using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MusicNotesProject.Migrations
{
    /// <inheritdoc />
    public partial class RenameFullPathToPath : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "FullPath",
                table: "Notes",
                newName: "Path");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "Path",
                table: "Notes",
                newName: "FullPath");
        }
    }
}
