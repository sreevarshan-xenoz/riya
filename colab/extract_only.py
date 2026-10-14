"""Unpack the latest /content/jarvis.zip over /content/code (running jobs keep their loaded code)."""
import zipfile

zipfile.ZipFile("/content/jarvis.zip").extractall("/content/code")
print("code updated")
