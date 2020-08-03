Set objArgs = WScript.Arguments
Set Rec = CreateObject("VisionRecorder.Application")
If Rec.CurrentWorkspace.Name <> objArgs(0) Then
  Rec.Acquisition.StopViewing()
  Rec.CurrentWorkspace.Load(objArgs(0))
End If
