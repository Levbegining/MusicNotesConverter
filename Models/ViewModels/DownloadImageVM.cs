using System;

namespace MusicNotesProject.Models.ViewModels;

public class ImageVM
{
    public List<IFormFile> Images { get; set; }
    public List<MusicNote> MusicNotes { get; set; }
}
