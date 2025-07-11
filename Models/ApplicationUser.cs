using System;
using Microsoft.AspNetCore.Identity;

namespace MusicNotesProject.Models;

public class ApplicationUser : IdentityUser
{
    public string FullName { get; set; }
}
