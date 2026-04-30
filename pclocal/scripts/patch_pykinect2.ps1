# Patch pykinect2 0.1.0 to work on 64-bit modern Python (3.10+).
#
# pykinect2 was last updated in 2017 and bakes in three assumptions that
# break on a fresh modern venv. All three are easy to fix with sed-like
# in-place edits — pykinect2's actual COM bindings still work; only the
# legacy guard rails fall over.
#
#   1. PyKinectV2.py has ~10 `assert sizeof(struct) == 32-bit-size` lines
#      that fail on 64-bit Python with messages like "AssertionError: 80".
#      The asserts are sanity checks only; the structs themselves use
#      ctypes.Structure auto-alignment, which is correct on 64-bit.
#
#   2. The auto-generated bindings call `_check_version('')` on import.
#      Any comtypes >= 1.2.0 treats the empty string as a version
#      mismatch and raises `ImportError("Wrong version")`. We don't need
#      that check (the bindings work fine against modern comtypes), so
#      we comment those calls out.
#
#   3. Older comtypes that *would* skip (2) (e.g. 1.1.10) use Python-2
#      `unicode` at module level and explode on Python 3.13 with
#      `NameError: name 'unicode' is not defined`. So we keep modern
#      comtypes and rely on the patch in (2) instead.
#
# Usage (from inside an activated venv):
#   .\scripts\patch_pykinect2.ps1
#
# Idempotent — re-running is safe; already-patched lines are skipped.

if (-not $env:VIRTUAL_ENV) {
    Write-Error "No active venv detected (`$env:VIRTUAL_ENV is empty)." `
        " Activate the pclocal venv before running this script."
    exit 1
}

$file = Join-Path $env:VIRTUAL_ENV "Lib\site-packages\pykinect2\PyKinectV2.py"
if (-not (Test-Path $file)) {
    Write-Error "pykinect2 not found at $file. " `
        "Did you run 'pip install -r requirements.txt'?"
    exit 1
}

# 1. Verify a modern comtypes is installed (we need Python-3-compatible
#    code, so 1.4.x is preferred). Older 1.1.x predates Python 3.12 and
#    still has Python 2 syntax in module top-level.
$comVerLine = pip show comtypes 2>$null | Select-String -Pattern '^Version:'
if ($null -eq $comVerLine) {
    Write-Error "comtypes is not installed. Run: pip install comtypes"
    exit 1
}
$comVer = ($comVerLine.ToString() -replace 'Version:\s*', '').Trim()
Write-Host "comtypes version: $comVer" -ForegroundColor DarkCyan
if ($comVer -match '^1\.1\.') {
    Write-Warning "comtypes 1.1.x is too old for Python 3.12+. " `
        "Run: pip install --force-reinstall comtypes"
}

# 2. Backup once.
$backup = "$file.bak"
if (-not (Test-Path $backup)) {
    Copy-Item $file $backup
    Write-Host "Backed up original to $backup"
}

# 3. Apply the two replacements. Both are idempotent — already-patched
#    lines start with `# patched ...` and don't match the patterns.
$content = Get-Content $file

# 3a. Comment out every `assert sizeof(...) == ...` line.
$content = $content -replace `
    '^(\s*)(assert\s+sizeof\([^)]+\)\s*==.*)$', `
    '$1# patched for 64-bit Python: $2'

# 3b. Comment out every `_check_version(...)` call (pykinect2 has these
#     scattered through the auto-generated bindings; modern comtypes
#     refuses empty version strings). The `(?!#)` lookahead skips
#     already-commented lines so the script stays idempotent.
$content = $content -replace `
    '^(\s*)(?!#)(.*\b_check_version\([^)]*\).*)$', `
    '$1# patched for modern comtypes: $2'

$content | Set-Content $file

$assertCount = (Select-String -Path $file -Pattern '# patched for 64-bit Python:').Count
$verCount    = (Select-String -Path $file -Pattern '# patched for modern comtypes:').Count
Write-Host "Patched assert lines:        $assertCount" -ForegroundColor Green
Write-Host "Patched _check_version lines: $verCount"  -ForegroundColor Green

# 4. Smoke test the import.
Write-Host "`nVerifying pykinect2 import..." -ForegroundColor Cyan
& python -c "from pykinect2 import PyKinectRuntime; print('pykinect2 OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "pykinect2 still fails to import. " `
        "Check the traceback above and rerun this script after fixing your venv."
    exit 1
}

Write-Host "`nDone. You can now run:  uvicorn app:app --host 127.0.0.1 --port 8000" -ForegroundColor Green
