import os
import sys
import importlib
import pkgutil
import customtkinter as ctk

# Configure CustomTkinter default appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class InstallationGuideModal(ctk.CTkToplevel):
    """Modern modal dialog showing step-by-step installation instructions and beginner code."""

    def __init__(self, parent, project_info, is_installed):
        super().__init__(parent)
        self.project_info = project_info
        self.is_installed = is_installed

        self.title(f"Installation & Setup Guide - {project_info['title']}")
        self.geometry("640x580")
        self.resizable(False, False)

        # Make modal window stay on top of parent
        self.transient(parent)
        self.grab_set()

        self.setup_ui()

    def setup_ui(self):
        self.configure(fg_color=("#EFF1F5", "#181825"))

        # Top Header Banner
        header = ctk.CTkFrame(
            self,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=0,
            height=70
        )
        header.pack(fill="x")

        icon_lbl = ctk.CTkLabel(
            header,
            text=f"{self.project_info['icon']}  {self.project_info['title']}",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC")
        )
        icon_lbl.pack(side="left", padx=20, pady=18)

        status_text = "Installed ✓" if self.is_installed else "Setup Required ⚠️"
        status_bg = ("#40A02B", "#A6E3A1") if self.is_installed else ("#FE640B", "#FAB387")
        status_pill = ctk.CTkLabel(
            header,
            text=status_text,
            fg_color=status_bg,
            text_color=("#FFFFFF", "#11111B"),
            corner_radius=8,
            padx=12,
            pady=4,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        status_pill.pack(side="right", padx=20, pady=18)

        # Scrollable Guide Content
        guide_scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=("#CBD5E1", "#2B2C3B"),
            scrollbar_button_hover_color=("#94A3B8", "#3E4052")
        )
        guide_scroll.pack(fill="both", expand=True, padx=20, pady=15)

        # Requirements & Terminal Command Box
        cmd_box = ctk.CTkFrame(
            guide_scroll,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=10
        )
        cmd_box.pack(fill="x", pady=(0, 15))

        cmd_title = ctk.CTkLabel(
            cmd_box,
            text="💻 Required Python Packages",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        cmd_title.pack(anchor="w", padx=15, pady=(10, 5))

        cmd_row = ctk.CTkFrame(
            cmd_box,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=8
        )
        cmd_row.pack(fill="x", padx=15, pady=(0, 12))

        cmd_code = ctk.CTkLabel(
            cmd_row,
            text=self.project_info.get("install_command", "No install command specified."),
            font=ctk.CTkFont(family="Consolas", size=13),
            text_color=("#40A02B", "#A6E3A1")
        )
        cmd_code.pack(side="left", padx=15, pady=10)

        self.copy_btn = ctk.CTkButton(
            cmd_row,
            text="📋 Copy Command",
            width=110,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.copy_command
        )
        self.copy_btn.pack(side="right", padx=10, pady=6)

        # Full Guide Text Area
        guide_textbox = ctk.CTkTextbox(
            guide_scroll,
            height=320,
            fg_color=("#FFFFFF", "#1E1E2E"),
            text_color=("#1E1E2E", "#CDD6F4"),
            scrollbar_button_color=("#CBD5E1", "#2B2C3B"),
            scrollbar_button_hover_color=("#94A3B8", "#3E4052"),
            font=ctk.CTkFont(size=13),
            border_width=0
        )
        guide_textbox.pack(fill="both", expand=True)

        guide_content = self.project_info.get("guide", "No guide available.")
        guide_textbox.insert("1.0", guide_content)
        guide_textbox.configure(state="disabled")

        # Bottom Close Button
        bottom_frame = ctk.CTkFrame(
            self,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=0,
            height=50
        )
        bottom_frame.pack(fill="x")

        close_btn = ctk.CTkButton(
            bottom_frame,
            text="Close Guide",
            width=120,
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(weight="bold"),
            command=self.destroy
        )
        close_btn.pack(side="right", padx=20, pady=10)

    def copy_command(self):
        cmd = self.project_info.get("install_command", "")
        self.clipboard_clear()
        self.clipboard_append(cmd)
        self.copy_btn.configure(
            text="Copied! ✓",
            fg_color=("#40A02B", "#A6E3A1"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.after(
            2000,
            lambda: self.copy_btn.configure(
                text="📋 Copy Command",
                fg_color=("#DCE0E8", "#313244"),
                text_color=("#4C4F69", "#CDD6F4")
            )
        )


class ProjectCard(ctk.CTkFrame):
    """Custom Card widget representing an AI Project with full Light/Dark support."""

    def __init__(self, parent, project_data, on_view_callback, on_info_callback):
        super().__init__(
            parent,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=14,
            border_width=1,
            border_color=("#CCD0DA", "#313244")
        )
        self.project_data = project_data
        self.info = project_data["info"]
        self.is_installed = project_data["is_installed"]
        self.on_view_callback = on_view_callback
        self.on_info_callback = on_info_callback

        self.setup_card_ui()

    def setup_card_ui(self):
        # 1. Action Buttons Bar (Anchored to Bottom of Card)
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(side="bottom", fill="x", padx=16, pady=(0, 16))

        # View / Open Button
        view_btn_text = "Launch Project ▶" if self.is_installed else "View Setup ⚙"
        view_btn_color = ("#1E66F5", "#89B4FA") if self.is_installed else ("#FE640B", "#FAB387")
        hover_color = ("#7287FD", "#B4BEFE") if self.is_installed else ("#E64553", "#F9E2AF")

        view_btn = ctk.CTkButton(
            btn_row,
            text=view_btn_text,
            fg_color=view_btn_color,
            hover_color=hover_color,
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34,
            command=lambda: self.on_view_callback(self.project_data)
        )
        view_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        # "ℹ" Info / Guide Button
        info_btn = ctk.CTkButton(
            btn_row,
            text="ℹ",
            width=36,
            height=34,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            font=ctk.CTkFont(size=14, weight="bold"),
            command=lambda: self.on_info_callback(self.project_data)
        )
        info_btn.pack(side="right")

        # 2. Requirements Tags (Anchored directly above Action Buttons)
        reqs = ", ".join(self.info.get("required_packages", []))
        req_lbl = ctk.CTkLabel(
            self,
            text=f"📦 Dependencies: {reqs}",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color=("#1E66F5", "#89B4FA"),
            justify="left",
            anchor="w"
        )
        req_lbl.pack(side="bottom", fill="x", padx=16, pady=(0, 10))

        # 3. Top Header (Icon + Status Badge) (Anchored to Top of Card)
        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(side="top", fill="x", padx=16, pady=(16, 6))

        icon_label = ctk.CTkLabel(
            top_row,
            text=self.info.get("icon", "🤖"),
            font=ctk.CTkFont(size=32)
        )
        icon_label.pack(side="left")

        status_text = "Ready to Run" if self.is_installed else "Setup Required"
        status_bg = ("#40A02B", "#A6E3A1") if self.is_installed else ("#FE640B", "#FAB387")
        status_badge = ctk.CTkLabel(
            top_row,
            text=status_text,
            fg_color=status_bg,
            text_color=("#FFFFFF", "#11111B"),
            corner_radius=6,
            padx=8,
            pady=3,
            font=ctk.CTkFont(size=11, weight="bold")
        )
        status_badge.pack(side="right")

        # 4. Project Title (Anchored below Header)
        title_lbl = ctk.CTkLabel(
            self,
            text=self.info.get("title", "Untitled Project"),
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC"),
            justify="left",
            anchor="w"
        )
        title_lbl.pack(side="top", fill="x", padx=16, pady=(0, 4))

        # 5. Project Description (Middle space expanding component)
        desc_lbl = ctk.CTkLabel(
            self,
            text=self.info.get("description", ""),
            font=ctk.CTkFont(size=12),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            anchor="nw"
        )
        desc_lbl.pack(side="top", fill="both", expand=True, padx=16, pady=(0, 6))

        def _update_wraplengths(event):
            try:
                scaling = ctk.ScalingTracker.get_widget_scaling(self)
            except Exception:
                scaling = 1.0
            
            # Account for display DPI scaling so text wraps safely inside card boundaries
            avail_w = max(140, int((event.width - 48) / scaling))
            title_lbl.configure(wraplength=avail_w)
            desc_lbl.configure(wraplength=avail_w)
            req_lbl.configure(wraplength=avail_w)

        self.bind("<Configure>", _update_wraplengths)


class AIPlayKitApp(ctk.CTk):
    """Main Application Window for AI PlayKit."""

    def __init__(self):
        super().__init__()

        self.title("AI PlayKit Hub")
        self.geometry("980x680")
        self.minsize(850, 550)
        self.configure(fg_color=("#EFF1F5", "#11111B"))
        self.bind("<1>", lambda event: event.widget.focus_set() if hasattr(event.widget, "focus_set") else None)
        self.VERSION = "1.0.0"

        self.projects = []
        self.active_project_frame = None
        self.hub_scroll = None
        self.app_logo_img = None
        self.welcome_logo_img = None

        self.load_app_logo()
        self.load_ai_projects()
        self.setup_main_ui()

    def load_app_logo(self):
        """Loads light & dark mode app logos from icons directory."""
        icons_dir = os.path.join(os.path.dirname(__file__), "icons")
        black_icon_path = os.path.join(icons_dir, "black.png")
        white_icon_path = os.path.join(icons_dir, "white.png")

        if os.path.exists(black_icon_path) and os.path.exists(white_icon_path):
            try:
                from PIL import Image, ImageTk
                img_black = Image.open(black_icon_path)
                img_white = Image.open(white_icon_path)

                self.app_logo_img = ctk.CTkImage(
                    light_image=img_black,
                    dark_image=img_white,
                    size=(34, 34)
                )
                self.welcome_logo_img = ctk.CTkImage(
                    light_image=img_black,
                    dark_image=img_white,
                    size=(54, 54)
                )

                # Set window taskbar icon
                window_icon = ImageTk.PhotoImage(img_black)
                self.iconphoto(False, window_icon)
            except Exception as e:
                print(f"Error loading logo icons: {e}")

    def load_ai_projects(self):
        """Dynamically imports and discovers project modules, sorted by date added (file creation time)."""
        self.projects = []
        projects_dir = os.path.join(os.path.dirname(__file__), "ai_modules")

        if not os.path.exists(projects_dir):
            return

        if os.path.dirname(__file__) not in sys.path:
            sys.path.insert(0, os.path.dirname(__file__))

        discovered = []
        for _, module_name, is_pkg in pkgutil.iter_modules([projects_dir]):
            if is_pkg or module_name.startswith("__"):
                continue

            filepath = os.path.join(projects_dir, f"{module_name}.py")
            created_time = os.path.getctime(filepath) if os.path.exists(filepath) else 0

            try:
                mod = importlib.import_module(f"ai_modules.{module_name}")
                if hasattr(mod, "PROJECT_INFO") and hasattr(mod, "check_dependencies"):
                    is_installed = mod.check_dependencies()
                    discovered.append((
                        created_time,
                        {
                            "module_name": module_name,
                            "module": mod,
                            "info": mod.PROJECT_INFO,
                            "is_installed": is_installed
                        }
                    ))
            except Exception as e:
                print(f"Error loading project module '{module_name}': {e}")

        # Sort by creation time (order of date added)
        discovered.sort(key=lambda item: item[0])
        self.projects = [item[1] for item in discovered]

    def setup_main_ui(self):
        # Navigation / App Header
        self.header_frame = ctk.CTkFrame(
            self,
            fg_color=("#FFFFFF", "#181825"),
            corner_radius=0,
            height=70
        )
        self.header_frame.pack(fill="x")

        logo_lbl = ctk.CTkLabel(
            self.header_frame,
            text="🎓 Codénix AI PlayKit Hub",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        logo_lbl.pack(side="left", padx=25, pady=15)

        sub_lbl = ctk.CTkLabel(
            self.header_frame,
            text="Interactive AI Projects for Beginners",
            font=ctk.CTkFont(size=13),
            text_color=("#5C5F77", "#A6ADC8")
        )
        sub_lbl.pack(side="left", padx=(0, 20), pady=15)

        # About / Info Button (Rightmost next to search bar)
        self.about_btn = ctk.CTkButton(
            self.header_frame,
            text="ℹ",
            width=34,
            height=32,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.show_about_dialog
        )
        self.about_btn.pack(side="right", padx=(0, 25), pady=18)

        # Search Bar
        self.search_entry = ctk.CTkEntry(
            self.header_frame,
            placeholder_text="🔍 Search AI projects...",
            width=220,
            fg_color=("#F2F4F8", "#11111B"),
            text_color=("#1E1E2E", "#CDD6F4"),
            font=ctk.CTkFont(size=12)
        )
        self.search_entry.pack(side="right", padx=(0, 8), pady=18)
        self.search_entry.bind("<KeyRelease>", self.filter_projects)

        # Theme Switcher Button (Light / Dark)
        self.theme_btn = ctk.CTkButton(
            self.header_frame,
            text="🌙 Dark Theme",
            width=110,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.toggle_theme
        )
        self.theme_btn.pack(side="right", padx=(0, 8), pady=18)

        # Main Workspace Container
        self.workspace_container = ctk.CTkFrame(self, fg_color="transparent")
        self.workspace_container.pack(fill="both", expand=True)

        self.setup_hub_view()

    def show_about_dialog(self):
        """Displays simple messagebox with developer & author details."""
        from tkinter import messagebox
        messagebox.showinfo(
            "About",
            "🎓 Codénix AI PlayKit Hub\n\n"
            "Developed by: Codénix Coding Club\n"
            "Organisation: Girijananda Chowdhury University"
            "Authors: Akash Bora\n\n"
            "An interactive suite of AI applications designed for learning and exploration.\n"
            "Version: "+self.VERSION
        )

    def toggle_theme(self):
        current = ctk.get_appearance_mode()
        if current == "Dark":
            ctk.set_appearance_mode("Light")
            self.theme_btn.configure(text="☀️ Light Theme")
        else:
            ctk.set_appearance_mode("Dark")
            self.theme_btn.configure(text="🌙 Dark Theme")

    def setup_hub_view(self):
        """Creates the permanent scrollable Hub view container ONCE during app setup."""
        self.hub_scroll = ctk.CTkScrollableFrame(
            self.workspace_container,
            fg_color="transparent",
            scrollbar_button_color=("#CBD5E1", "#2B2C3B"),
            scrollbar_button_hover_color=("#94A3B8", "#3E4052")
        )
        self.hub_scroll.pack(fill="both", expand=True, padx=15, pady=15)

        # Welcome Section
        welcome_box = ctk.CTkFrame(
            self.hub_scroll,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12
        )
        welcome_box.pack(fill="x", pady=(0, 15))

        welcome_inner = ctk.CTkFrame(welcome_box, fg_color="transparent")
        welcome_inner.pack(fill="x", padx=20, pady=16)

        # Left Column: Prominent Logo Image (54x54) spanning both title & description
        if self.welcome_logo_img:
            logo_lbl = ctk.CTkLabel(
                welcome_inner,
                image=self.welcome_logo_img,
                text=""
            )
            logo_lbl.pack(side="left", padx=(0, 18), anchor="center")

        # Right Column: Stacked Title and Description Labels
        text_col = ctk.CTkFrame(welcome_inner, fg_color="transparent")
        text_col.pack(side="left", fill="both", expand=True)

        w_title = ctk.CTkLabel(
            text_col,
            text="Explore & Build AI Applications",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC"),
            anchor="w"
        )
        w_title.pack(fill="x", anchor="w", pady=(0, 2))

        w_desc = ctk.CTkLabel(
            text_col,
            text="Click any project tile to launch its interactive interface. Optional dependencies will show setup guides automatically.",
            font=ctk.CTkFont(size=13),
            text_color=("#5C5F77", "#CDD6F4"),
            anchor="w",
            justify="left"
        )
        w_desc.pack(fill="x", anchor="w")

        # Cards Grid Frame - Pack fill="x" so cards sit right at the top without stretching
        self.grid_frame = ctk.CTkFrame(self.hub_scroll, fg_color="transparent")
        self.grid_frame.pack(fill="x", expand=False)

        self.render_project_cards(self.projects)

    def show_hub_view(self):
        """Displays the main Projects Cards Grid Hub instantly without destroying or rebuilding cards."""
        if self.active_project_frame:
            try:
                self.active_project_frame.destroy()
            except Exception as e:
                print(f"Error destroying active project frame: {e}")
            self.active_project_frame = None

        if self.hub_scroll:
            self.hub_scroll.pack(fill="both", expand=True, padx=15, pady=15)

    def render_project_cards(self, projects_list):
        for child in self.grid_frame.winfo_children():
            child.destroy()

        if not projects_list:
            no_results = ctk.CTkLabel(
                self.grid_frame,
                text="No matching projects found.",
                font=ctk.CTkFont(size=14),
                text_color=("#5C5F77", "#A6ADC8")
            )
            no_results.pack(pady=40)
            return

        col_count = 3
        row_count = (len(projects_list) + col_count - 1) // col_count
        for r in range(row_count):
            self.grid_frame.grid_rowconfigure(r, weight=1)
        for c in range(col_count):
            self.grid_frame.grid_columnconfigure(c, weight=1, uniform="card_col")

        for idx, project_data in enumerate(projects_list):
            row = idx // col_count
            col = idx % col_count

            card = ProjectCard(
                self.grid_frame,
                project_data=project_data,
                on_view_callback=self.open_project_view,
                on_info_callback=self.open_info_modal
            )
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

    def filter_projects(self, event=None):
        query = self.search_entry.get().strip().lower()
        if not query:
            self.render_project_cards(self.projects)
            return

        filtered = [
            p for p in self.projects
            if query in p["info"]["title"].lower() or query in p["info"]["description"].lower()
        ]
        self.render_project_cards(filtered)

    def open_project_view(self, project_data):
        """Handles View button click. Hides Hub and launches module UI instantly."""
        if not project_data["is_installed"]:
            self.open_info_modal(project_data)
            return

        # Hide Hub scroll view instantly without destroying widgets
        if self.hub_scroll:
            self.hub_scroll.pack_forget()

        if self.active_project_frame:
            try:
                self.active_project_frame.destroy()
            except Exception:
                pass
            self.active_project_frame = None

        # Instantiate UI class from project module
        mod = project_data["module"]
        
        ui_class = None
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, ctk.CTkFrame) and attr != ctk.CTkFrame:
                ui_class = attr
                break

        if ui_class:
            self.active_project_frame = ui_class(self.workspace_container, on_back_callback=self.show_hub_view)
            self.active_project_frame.pack(fill="both", expand=True, padx=10, pady=10)
        else:
            print("No CustomTkinter Frame class found in module.")

    def open_info_modal(self, project_data):
        """Displays full installation and code guide in a modal dialog."""
        InstallationGuideModal(self, project_data["info"], project_data["is_installed"])


if __name__ == "__main__":
    app = AIPlayKitApp()
    app.mainloop()
