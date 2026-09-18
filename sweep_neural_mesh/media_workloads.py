"""Bounded media workloads using existing SWEEP vision and speech components."""
from __future__ import annotations
import base64,os,wave
from pathlib import Path
from typing import Any
from .workloads import _file,_sha

def _path(path:str):
 try:return _file(path)
 except ValueError as exc:return None,{"status":"error","path":str(path),"reason":str(exc)}
def _unavailable(path:Path,reason:str)->dict[str,Any]:return {"status":"unavailable","path":str(path),"sha256":_sha(path),"reason":reason}
def image_run(path:str,operation:str="describe",output:str|None=None,params:dict[str,Any]|None=None)->dict[str,Any]:
 p,error=_path(path)
 if p is None:return error
 params=dict(params or {});operation=operation.lower()
 try:from companion.tools.vision import run_vision
 except Exception as exc:return _unavailable(p,f"Vision adapter unavailable: {type(exc).__name__}")
 try:encoded=base64.b64encode(p.read_bytes()).decode("ascii");result=run_vision({"image_base64":encoded,"params":{"operation":operation,**params}})
 except Exception as exc:return _unavailable(p,f"Vision execution unavailable: {type(exc).__name__}: {str(exc)[:160]}")
 result={"status":"completed","path":str(p),"sha256":_sha(p),"operation":operation,**result};image_b64=(result.get("result") or {}).get("image_base64")
 if image_b64:
  if not output:return {"status":"error","path":str(p),"sha256":_sha(p),"reason":"An output path is required for image-transform operations."}
  out=Path(output).expanduser().resolve()
  if out==p:raise ValueError("Output path must differ from input path.")
  out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(base64.b64decode(image_b64));result["output"]={"path":str(out),"sha256":_sha(out),"bytes":out.stat().st_size};del result["result"]["image_base64"]
 return result
def audio_inspect(path:str)->dict[str,Any]:
 p,error=_path(path)
 if p is None:return error
 ext=p.suffix.lower();result={"status":"completed","path":str(p),"sha256":_sha(p),"bytes":p.stat().st_size,"suffix":ext}
 if ext==".wav":
  try:
   with wave.open(str(p),"rb") as stream:result.update({"format":"wav","channels":stream.getnchannels(),"sample_width_bytes":stream.getsampwidth(),"sample_rate_hz":stream.getframerate(),"frames":stream.getnframes(),"duration_seconds":round(stream.getnframes()/max(1,stream.getframerate()),4)})
  except wave.Error as exc:return {"status":"error","path":str(p),"sha256":_sha(p),"reason":f"Invalid WAV: {exc}"}
 else:result.update({"format":"unparsed","supported_transcription_extensions":[".wav",".mp3",".flac",".ogg",".m4a",".opus",".webm"],"note":"Codec metadata requires optional audio tooling."})
 return result
def audio_transcribe(path:str)->dict[str,Any]:
 p,error=_path(path)
 if p is None:return error
 cache=Path(os.environ.get("SWEEP_WHISPER_MODEL_PATH",Path.home()/".cache/whisper/tiny.en.pt")).expanduser()
 if not cache.exists():return _unavailable(p,"Local Whisper tiny.en weights not found; install or place them explicitly before transcription. No download was attempted.")
 try:
  from sweep_neural_mesh.neurons.speech_recognition import SpeechRecognizer
  result=SpeechRecognizer().recognize(str(p));return {"status":"completed" if result.backend!="none" else "unavailable","path":str(p),"sha256":_sha(p),"backend":result.backend,"text":result.text,"language":result.language,"duration_seconds":result.duration_seconds,"latency_ms":result.latency_ms,"segments":[{"start":s.start,"end":s.end,"text":s.text} for s in result.segments]}
 except Exception as exc:return _unavailable(p,f"Audio transcription unavailable: {type(exc).__name__}: {str(exc)[:160]}")
def video_inspect(path:str)->dict[str,Any]:
 p,error=_path(path)
 if p is None:return error
 try:import cv2
 except Exception as exc:return _unavailable(p,f"Video inspection requires OpenCV: {type(exc).__name__}")
 cap=cv2.VideoCapture(str(p))
 if not cap.isOpened():return {"status":"error","path":str(p),"sha256":_sha(p),"reason":"OpenCV could not open this video or codec is unavailable."}
 try:
  frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));duration=frames/fps if fps>0 else None
  return {"status":"completed","path":str(p),"sha256":_sha(p),"bytes":p.stat().st_size,"frames":frames,"fps":fps,"width":width,"height":height,"duration_seconds":duration}
 finally:cap.release()
