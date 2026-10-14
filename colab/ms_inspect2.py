import inspect
import moonshine_voice as mv
import moonshine_voice.download as dl
import moonshine_voice.transcriber as tr
import moonshine_onnx
print("==== Stream\n", inspect.getsource(tr.Stream)[:9000])
print("==== find_model_info\n", inspect.getsource(dl.find_model_info))
src = inspect.getsource(dl)
i = src.find('"en"')
print("==== download.py around en\n", src[max(0, i - 1500):i + 2500])
print("==== onnx load_audio\n", inspect.getsource(moonshine_onnx.load_audio) if hasattr(moonshine_onnx, "load_audio") else "none")
print("==== onnx transcribe\n", inspect.getsource(moonshine_onnx.transcribe))
print("onnx attrs", [k for k in dir(moonshine_onnx) if not k.startswith("_")])
