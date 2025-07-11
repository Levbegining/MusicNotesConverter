using System;
using MusicNotesProject.Data;

namespace MusicNotesProject.Models;

public class MusicNote
{
    public int Id { get; set; }
    /// <summary>
    /// Имя файла изображения без расширения
    /// </summary>
    public string Name { get; set; }
    public string Path { get; set; } = Constants.FullImagePath;
}
