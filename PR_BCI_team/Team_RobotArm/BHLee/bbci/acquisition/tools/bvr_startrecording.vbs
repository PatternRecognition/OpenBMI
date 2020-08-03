Set objArgs = WScript.Arguments
Set Rec = CreateObject("VisionRecorder.Application")
Rec.Acquisition.ViewData()
Rec.Acquisition.StartRecording(objArgs(0))
