"""Bounded local Windows OCR using the operating system's installed languages."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

_SCRIPT = r'''
param([string]$ImagePath, [string]$OutputPath)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.IRandomAccessStreamWithContentType, Windows.Storage.Streams, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null
$method = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1
} | Select-Object -First 1
function Await-Result($Operation, [Type]$ResultType) {
    $task = $method.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}
$file = Await-Result ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
$stream = Await-Result ($file.OpenReadAsync()) ([Windows.Storage.Streams.IRandomAccessStreamWithContentType])
try {
    $decoder = Await-Result ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await-Result ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    try {
        $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
        if ($null -eq $engine) { throw 'No Windows OCR language is installed.' }
        $result = Await-Result ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        $text = ($result.Lines | ForEach-Object { $_.Text }) -join "`n"
        [System.IO.File]::WriteAllText($OutputPath, $text, [System.Text.UTF8Encoding]::new($false))
    } finally { if ($bitmap) { $bitmap.Dispose() } }
} finally { $stream.Dispose() }
'''


def windows_ocr(image) -> dict:
    from .images import MAX_OCR_TEXT, _render_copy
    engine = "Windows OCR (local)"
    if os.name != "nt":
        return {"status": "unavailable", "engine": engine, "text": "", "message": "Windows OCR is unavailable on this system."}
    executable = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    try:
        with tempfile.TemporaryDirectory(prefix="sweep-ocr-") as directory:
            root = Path(directory)
            script, source, output = root / "ocr.ps1", root / "image.png", root / "text.txt"
            script.write_text(_SCRIPT, encoding="utf-8-sig")
            with _render_copy(image, 2500) as copy:
                copy.save(source, format="PNG")
            subprocess.run([str(executable), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                            "-File", str(script), str(source), str(output)],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=20, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with output.open(encoding="utf-8") as stream:
                text = stream.read(MAX_OCR_TEXT + 1)
        return {"status": "completed", "engine": engine, "text": text[:MAX_OCR_TEXT], "truncated": len(text) > MAX_OCR_TEXT,
                "message": "Text was read locally using Windows OCR; recognition may contain errors."}
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable", "engine": engine, "text": "", "message": "Local OCR could not run. Install a supported Windows OCR language or Tesseract; metadata is still available."}
