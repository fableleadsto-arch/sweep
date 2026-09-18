import tempfile,unittest,wave
from pathlib import Path
from sweep_neural_mesh.media_workloads import audio_inspect,audio_transcribe,image_run
class MediaWorkloadTests(unittest.TestCase):
 def test_wav_metadata_is_real(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,"tone.wav")
   with wave.open(str(p),"wb") as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000);w.writeframes(b"\0\0"*8000)
   out=audio_inspect(str(p));self.assertEqual(out["status"],"completed");self.assertEqual(out["duration_seconds"],1.0);self.assertEqual(len(out["sha256"]),64)
 def test_missing_whisper_is_explicit(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,"audio.wav");p.write_bytes(b"not-audio");out=audio_transcribe(str(p));self.assertEqual(out["status"],"unavailable");self.assertIn("No download",out["reason"])
 def test_image_missing_is_explicit(self):
  out=image_run("/tmp/sweep-no-such-image.png");self.assertEqual(out["status"],"error")
if __name__=="__main__":unittest.main()
