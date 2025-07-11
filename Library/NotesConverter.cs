using System;
using System.Diagnostics;

namespace MusicNotesProject.Library;

public class NotesConverter
{
    public static void ConvertToMidi(List<string>? fileNames = null, string pythonScriptPath =
    @"/Users/lev/Projects/Asp/MusicNotesProject/Library/PythonVenv/python_script.py", string mergeImagePath = @"/Users/lev/Projects/Asp/MusicNotesProject/Library/PythonVenv/merge_img.py",
    bool MergeImage = false)
    {
        // const string mergeImagePath = @"/Users/lev/Projects/Asp/MusicNotesProject/Library/PythonVenv/megre_img.py";
        const string pathToVenvPython = "/Users/lev/Projects/Asp/MusicNotesProject/Library/PythonVenv/.venv/bin/python";
        if (MergeImage)
        {
            if (fileNames != null && fileNames.Count != 0)
            {
                var escapedPaths = fileNames.Select(path => $"'{path}'").ToList();
                string fileNamesString = string.Join(" ", escapedPaths);
                var args = $"{mergeImagePath} --args \"{fileNamesString}\"";
                DoProccess(args, pathToVenvPython);
            }
            else
            {
                DoProccess($"{mergeImagePath}", pathToVenvPython);
            }
            
            DoProccess($"{pythonScriptPath}", pathToVenvPython);
        }
        else
        {
            DoProccess($"{pythonScriptPath}", pathToVenvPython);
        }

    }
    private static void DoProccess(string args, string pathToVenvPython = "/Users/lev/Projects/Asp/MusicNotesProject/Library/PythonVenv/.venv/bin/python")
    {
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = pathToVenvPython, // или "python3" в Linux/macOS
                Arguments = args,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = "/Users/lev/Projects/Asp/MusicNotesProject/Library/PythonVenv"
            }
        };

        process.Start();
        // string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (!string.IsNullOrEmpty(error))
            throw new Exception($"Python error: {error}");
    }
}