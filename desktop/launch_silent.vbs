Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
strDesktopDir = fso.GetParentFolderName(WScript.ScriptFullName)
strProjectRoot = fso.GetParentFolderName(strDesktopDir)

WshShell.CurrentDirectory = strProjectRoot

' Launch pythonw.exe completely hidden (0 = hide console window)
pythonwPath = "C:\Users\sumuk\AppData\Local\Programs\Python\Python311\pythonw.exe"
If fso.FileExists(pythonwPath) Then
    WshShell.Run """" & pythonwPath & """ run_desktop_app.py", 0, False
Else
    WshShell.Run "pythonw.exe run_desktop_app.py", 0, False
End If
