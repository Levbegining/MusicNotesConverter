using System;
using Microsoft.AspNetCore.Mvc;
using MusicNotesProject.Data;
using MusicNotesProject.Library;
using MusicNotesProject.Models;
using MusicNotesProject.Models.ViewModels;

namespace MusicNotesProject.Controllers;

public class MusicNoteController : Controller
{
    private readonly ApplicationDbContext context;
    public MusicNoteController(ApplicationDbContext context)
    {
        this.context = context;
    }

    // private List<DownloadImageVM> model;
    [HttpPost]
    public IActionResult UploadImages(ImageVM model)
    {
        var path = $"{Constants.FullImagePathWithoutName}/MusicNote";

        // для БД
        var lastIdContext = context.Notes.OrderByDescending(n => n.Id).FirstOrDefault()?.Id ?? 0;
        var musicNotes = context.Notes;

        for (int i = 0; i < model.Images.Count; i++)
        {
            var file = model.Images[i];
            var fileExt = file.FileName.Split('.').Last();
            // Проверяем, что файл не пустой
            if (file.Length > 0)
            {
                lastIdContext++;
                var fullPath = $"{path}_{lastIdContext}.{fileExt}";
                var _path = $"/external-images/";
                // Сохраняем файл на диск
                using (var fileStream = new FileStream(fullPath, FileMode.Create))
                {
                    file.CopyTo(fileStream);
                }

                // сохранение в БД
                MusicNote musicNote = new MusicNote()
                {
                    Name = $"MusicNote_{lastIdContext}",
                    Path = Path.Combine(_path, $"MusicNote_{lastIdContext}.{fileExt}")
                };
                musicNotes.Add(musicNote);
            }
        }
        context.SaveChanges();

        return RedirectToAction(nameof(Index), new { messageUploadFiles = "Файлы успешно загружены!" });
    }
    public IActionResult Index(string messageUploadFiles = null, string messageConvertFiles = null)
    {
        if (messageUploadFiles != null)
        {
            ViewBag.MessageUploadFiles = messageUploadFiles;
        }
        if(messageConvertFiles != null)
        {
            ViewBag.MessageConvertFiles = messageConvertFiles;
        }
        var model = new ImageVM() { MusicNotes = context.Notes.ToList() };
        return View(model);
    }

    [HttpPost]
    public IActionResult ConvertToMidi(ImageVM model)
    {
        var path = $"{Constants.FullImagePathWithoutName}/MusicNote";

        // для БД
        var lastIdContext = context.Notes.OrderByDescending(n => n.Id).FirstOrDefault()?.Id ?? 0;
        var musicNotes = context.Notes;

        for (int i = 0; i < model.Images.Count; i++)
        {
            var file = model.Images[i];
            var fileExt = file.FileName.Split('.').Last();
            // Проверяем, что файл не пустой
            if (file.Length > 0)
            {
                lastIdContext++;
                var fullPath = $"{path}_{lastIdContext}.{fileExt}";
                var _path = $"/external-images/";
                // Сохраняем файл на диск
                using (var fileStream = new FileStream(fullPath, FileMode.Create))
                {
                    file.CopyTo(fileStream);
                }

                // сохранение в БД
                MusicNote musicNote = new MusicNote()
                {
                    Name = $"MusicNote_{lastIdContext}",
                    Path = Path.Combine(_path, $"MusicNote_{lastIdContext}.{fileExt}")
                };
                musicNotes.Add(musicNote);
            }
        }
        context.SaveChanges();



        // Здесь вы можете вызвать метод конвертации
        NotesConverter.ConvertToMidi(MergeImage: true);
        System.Console.WriteLine("Конвертация в MIDI завершена!");
        return RedirectToAction(nameof(Index));
    }

    [HttpPost]
    public IActionResult ConvertToMidiSomeNotes(List<string> selectedPaths)
    {
        if (selectedPaths[0] == "on")
        {
            selectedPaths.RemoveAt(0);
        }
        if (selectedPaths == null || selectedPaths.Count == 0)
        {
            return RedirectToAction(nameof(Index), new { messageConvertFiles = "Не выбраны файлы для конвертации!" });
        }
        // Здесь вы можете вызвать метод конвертации
        var fileList = selectedPaths.Select( path => path.Split('/').Last()).ToList();
        NotesConverter.ConvertToMidi(fileNames: fileList, MergeImage: true);
        System.Console.WriteLine("Конвертация в MIDI завершена!");

        byte[] fileBytes = System.IO.File.ReadAllBytes(Constants.FullMidiPath);
        return File(fileBytes, "application/octet-stream", "MusicNote.mid");
        // return RedirectToAction(nameof(Index));
    }
}
