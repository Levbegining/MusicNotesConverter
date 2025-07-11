using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity;
using MusicNotesProject.Models;
using MusicNotesProject.Data;
using Microsoft.Extensions.FileProviders;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();

builder.Services.AddAuthorization();
// вместо
// builder.Services.AddDefaultIdentity<IdentityUser>().
// AddEntityFrameworkStores<ApplicationDbContext>();
builder.Services.AddIdentity<IdentityUser, IdentityRole>()
    .AddDefaultTokenProviders()
    .AddDefaultUI()
    .AddEntityFrameworkStores<ApplicationDbContext>(); // для полного Identity с ролями

// для БД
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlite(builder.Configuration.
    GetConnectionString("MyDataBase"))
);


var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseRouting(); // 1. Включаем маршрутизацию
app.UseAuthentication(); // 2. Аутентификация (куки, JWT и т. д.)
app.UseAuthorization();
app.UseHttpsRedirection();
app.UseStaticFiles();
// Добавляем доступ к внешней папке
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(
        Path.Combine("/Users/lev/Projects/Asp/MusicNotesProject/Library/PythonVenv/Images")),
    RequestPath = "/external-images" // URL-префикс
});
//app.UseSession();
app.MapRazorPages(); // For Razor Page
app.MapControllerRoute(
name: "default",
pattern: "{controller=MusicNote}/{action=Index}/");

app.Run();
