using System;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using MusicNotesProject.Models;

namespace MusicNotesProject.Data;

public class ApplicationDbContext : IdentityDbContext<IdentityUser>
{
    public DbSet<MusicNote> Notes { get; set; }

    public DbSet<ApplicationUser> ApplicationUsers { get; set; }

    // Вариант 3
    public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options) : base(options) { }
}
