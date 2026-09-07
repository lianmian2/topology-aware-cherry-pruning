from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import List, Optional

import cv2
import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from skeleton_annotation.io_handler import (
        collect_image_files, load_image, load_mask, load_annotation, save_annotation,
    )
    from skeleton_annotation.renderer import CanvasRenderer
    from skeleton_annotation.snap import snap_to_existing_line, snap_to_existing_point
    from skeleton_annotation.state import AnnotationState
else:
    from .io_handler import (
        collect_image_files, load_image, load_mask, load_annotation, save_annotation,
    )
    from .renderer import CanvasRenderer
    from .snap import snap_to_existing_line, snap_to_existing_point
    from .state import AnnotationState


class SkeletonAnnotationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("骨架拓扑标注工具")
        self.root.geometry("1520x940")
        self.root.configure(bg="#2b2b2b")

        self._image_folder: Optional[Path] = None
        self._image_files: List[Path] = []
        self._current_index: int = -1
        self._mask_path: Optional[str] = None

        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.height: int = 0
        self.width: int = 0
        self.branch_mask: Optional[np.ndarray] = None

        self.state = AnnotationState()
        self.renderer: Optional[CanvasRenderer] = None

        self._show_mask = tk.BooleanVar(value=False)
        self._is_panning = False
        self._pan_start = (0, 0)
        self._new_branch_waiting_trunk = False
        self._undo_stack: List[dict] = []
        self._prev_active_before_branch: Optional[str] = None
        self._prev_waiting_before_branch: bool = False

        self._build_ui()
        self._bind_events()
        self.root.after(100, self._prompt_open_folder)

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg="#2b2b2b")
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main, bg="#2b2b2b", width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        left.pack_propagate(False)
        self._build_toolbar(left)

        right = tk.Frame(main, bg="#1e1e1e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)

        self.canvas = tk.Canvas(right, bg="#1e1e1e", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(
            self.root, text="请先选择一个图片文件夹", anchor="w",
            bg="#2b2b2b", fg="#aaaaaa", font=("Microsoft YaHei", 9),
            padx=12, pady=4,
        )
        self.status.pack(fill=tk.X)

    def _build_toolbar(self, parent: tk.Frame) -> None:
        sty = {"bg": "#2b2b2b", "fg": "#e0e0e0", "font": ("Microsoft YaHei", 9)}
        btn = {
            "bg": "#3c3c3c", "fg": "#e0e0e0", "font": ("Microsoft YaHei", 9),
            "activebackground": "#505050", "activeforeground": "#ffffff",
            "relief": tk.FLAT, "bd": 0, "padx": 10, "pady": 5,
        }
        abtn = {**btn, "bg": "#0e639c", "activebackground": "#1177bb"}
        wbtn = {**btn, "bg": "#8b3a3a", "activebackground": "#a04545"}

        tk.Label(parent, text="文件", **sty).pack(anchor="w", pady=(0, 4))
        tk.Button(parent, text="📁 打开图片文件夹", command=self._prompt_open_folder,
                  **abtn).pack(fill=tk.X, pady=2)

        tk.Label(parent, text="图片列表", **sty).pack(anchor="w", pady=(8, 4))
        lf = tk.Frame(parent, bg="#1e1e1e", height=140)
        lf.pack(fill=tk.X, pady=2)
        lf.pack_propagate(False)
        self.image_listbox = tk.Listbox(
            lf, bg="#1e1e1e", fg="#e0e0e0", font=("Consolas", 9),
            selectbackground="#0e639c", selectforeground="#ffffff",
            relief=tk.FLAT, bd=0, highlightthickness=0,
        )
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        s1 = tk.Scrollbar(lf, bg="#2b2b2b", troughcolor="#1e1e1e")
        s1.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.config(yscrollcommand=s1.set)
        s1.config(command=self.image_listbox.yview)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_select)

        tk.Button(parent, text="🗺 加载分割Mask", command=self._prompt_load_mask,
                  **btn).pack(fill=tk.X, pady=(4, 2))
        tk.Checkbutton(
            parent, text="显示分割Mask", variable=self._show_mask,
            command=self._redraw, bg="#2b2b2b", fg="#e0e0e0",
            selectcolor="#2b2b2b", activebackground="#2b2b2b",
            activeforeground="#e0e0e0", font=("Microsoft YaHei", 9),
        ).pack(anchor="w", pady=4)

        tk.Label(parent, text="标注模式", **sty).pack(anchor="w", pady=(12, 4))
        tk.Button(parent, text="🌳 主干模式", command=self._mode_trunk,
                  **abtn).pack(fill=tk.X, pady=2)
        tk.Button(parent, text="🌿 新建枝条（起点必须在主干上）",
                  command=self._mode_new_branch, **btn).pack(fill=tk.X, pady=2)

        tk.Label(parent, text="标注分组", **sty).pack(anchor="w", pady=(12, 4))
        gf = tk.Frame(parent, bg="#1e1e1e", height=140)
        gf.pack(fill=tk.X, pady=2)
        gf.pack_propagate(False)
        self.group_listbox = tk.Listbox(
            gf, bg="#1e1e1e", fg="#e0e0e0", font=("Consolas", 9),
            selectbackground="#0e639c", selectforeground="#ffffff",
            relief=tk.FLAT, bd=0, highlightthickness=0,
        )
        self.group_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        s2 = tk.Scrollbar(gf, bg="#2b2b2b", troughcolor="#1e1e1e")
        s2.pack(side=tk.RIGHT, fill=tk.Y)
        self.group_listbox.config(yscrollcommand=s2.set)
        s2.config(command=self.group_listbox.yview)
        self.group_listbox.bind("<<ListboxSelect>>", self._on_group_select)

        tk.Label(parent, text="编辑", **sty).pack(anchor="w", pady=(8, 4))
        tk.Button(parent, text="↩ 撤销上一个点 (Ctrl+Z)",
                  command=self._undo_last, **btn).pack(fill=tk.X, pady=2)
        tk.Button(parent, text="🗑 删除当前组",
                  command=self._delete_group, **wbtn).pack(fill=tk.X, pady=2)
        tk.Button(parent, text="🗑 清空全部标注",
                  command=self._clear_all, **wbtn).pack(fill=tk.X, pady=2)

        tk.Label(parent, text="保存", **sty).pack(anchor="w", pady=(12, 4))
        tk.Button(parent, text="💾 手动保存当前标注",
                  command=self._manual_save, **abtn).pack(fill=tk.X, pady=2)

        tips = (
            "操作说明:\n"
            "1. 打开图片文件夹，左侧列表切换图片\n"
            "   （自动保存+自动加载已有标注）\n"
            "2. 主干模式: 左键点选主干关键点\n"
            "3. 新建枝条: 必须在主干线段上点起点\n"
            "   此后随时hover已有节点→放大反馈\n"
            "   →点击=从该节点分叉\n"
            "4. Ctrl+Z 撤销 | Ctrl+A 新建枝条\n"
            "   滚轮缩放, 右键平移\n"
            "5. 分叉点由拓扑自动推导（度≥3）\n"
            "   有几个枝条组=主干上有几个分叉点"
        )
        tk.Label(parent, text=tips, justify=tk.LEFT, wraplength=270,
                 bg="#2b2b2b", fg="#888888",
                 font=("Microsoft YaHei", 8)).pack(anchor="w", pady=(16, 0))

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Button-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.root.bind("<Control-z>", self._on_ctrl_z)
        self.root.bind("<Control-Z>", self._on_ctrl_z)
        self.root.bind("<Control-a>", self._on_ctrl_a)
        self.root.bind("<Control-A>", self._on_ctrl_a)

    def _prompt_open_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        self._image_folder = Path(folder)
        self._image_files = collect_image_files(self._image_folder)
        if not self._image_files:
            messagebox.showinfo("提示", "该文件夹中没有图片文件")
            return
        self._refresh_image_list()
        self._switch_to_image(0)

    def _refresh_image_list(self) -> None:
        self.image_listbox.delete(0, tk.END)
        for f in self._image_files:
            self.image_listbox.insert(tk.END, f.name)

    def _on_image_select(self, event: tk.Event) -> None:
        sel = self.image_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx != self._current_index:
            self._switch_to_image(idx)

    def _switch_to_image(self, index: int) -> None:
        if index < 0 or index >= len(self._image_files):
            return
        if self._current_index >= 0 and self.image_bgr is not None:
            self._save_current(silent=True)
        self._current_index = index
        file_path = self._image_files[index]
        self.image_bgr = load_image(file_path)
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image_rgb.shape[:2]
        self.branch_mask = load_mask(self._mask_path, self.height, self.width)
        self.state.clear()
        self._undo_stack.clear()
        self._new_branch_waiting_trunk = False
        ok, session = load_annotation(self.state, self._image_folder, file_path.stem)
        self.renderer = CanvasRenderer(
            self.canvas, self.image_rgb, self.width, self.height,
            self.branch_mask, self._show_mask.get(),
        )
        self._restore_session(session)
        self._refresh_image_list_selection()
        self._refresh_group_list()
        self._redraw()
        text = f"[{index+1}/{len(self._image_files)}] {file_path.name} | {self.width}x{self.height}"
        if ok:
            text += " | 已加载已有标注"
        self.status.config(text=text)

    def _refresh_image_list_selection(self) -> None:
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self._current_index)
        self.image_listbox.activate(self._current_index)
        self.image_listbox.see(self._current_index)

    def _prompt_load_mask(self) -> None:
        if self.image_bgr is None:
            messagebox.showinfo("提示", "请先打开图片文件夹并选择一张图片")
            return
        fp = filedialog.askopenfilename(
            title="选择分割Mask图片（二值图）",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")],
        )
        if not fp:
            return
        self._mask_path = fp
        self.branch_mask = load_mask(self._mask_path, self.height, self.width)
        if self.branch_mask is not None:
            self._show_mask.set(True)
            self.renderer.branch_mask = self.branch_mask
            self.renderer.show_mask = True
            self.status.config(text=f"已加载分割Mask: {Path(fp).name}")
        self._redraw()

    def _mode_trunk(self) -> None:
        gid = self.state.ensure_trunk_group()
        self.state.active_group_id = gid
        self._new_branch_waiting_trunk = False
        self._undo_stack.clear()
        self._clear_hover()
        self._sync_renderer_flags()
        self._refresh_group_list()
        self._redraw()
        self.status.config(text="主干模式 — 点击图片添加主干关键点")

    def _mode_new_branch(self) -> None:
        trunk = self.state.groups.get("trunk")
        if trunk is None or len(trunk.points) < 2:
            messagebox.showinfo("提示", "请先标注主干（至少2个点），再新建枝条")
            return
        self._prev_active_before_branch = self.state.active_group_id
        self._prev_waiting_before_branch = self._new_branch_waiting_trunk
        gid = self.state.add_branch_group()
        self._undo_stack.append({
            "type": "new_branch",
            "gid": gid,
            "prev_active": self._prev_active_before_branch,
            "prev_waiting": self._prev_waiting_before_branch,
        })
        self.state.active_group_id = gid
        self._new_branch_waiting_trunk = True
        self._clear_hover()
        self._sync_renderer_flags()
        self._refresh_group_list()
        self._redraw()
        self.status.config(text=f"新建枝条 '{gid}' — 请在主干线段上点击起点（青色高亮）")
        self._save_current(silent=True)

    def _sync_renderer_flags(self) -> None:
        if self.renderer is None:
            return
        self.renderer._new_branch_waiting = self._new_branch_waiting_trunk

    def _clear_hover(self) -> None:
        if self.renderer is not None:
            self.renderer._hovered_image_point = None

    def _is_branch_active(self) -> bool:
        gid = self.state.active_group_id
        if gid is None or gid not in self.state.groups:
            return False
        return self.state.groups[gid].group_type == "branch"

    def _on_mouse_move(self, event: tk.Event) -> None:
        if self.renderer is None:
            return

        if not self._is_branch_active() and not self._new_branch_waiting_trunk:
            if self.renderer._hovered_image_point is not None:
                self._clear_hover()
                self._redraw()
            return

        point = self.renderer.canvas_to_image_xy(event.x, event.y)
        if point is None:
            return

        target_gid = self.state.active_group_id if self._is_branch_active() else None
        exclude_first = bool(
            target_gid
            and target_gid in self.state.groups
            and self.state.groups[target_gid].group_type == "branch"
        )
        snapped = snap_to_existing_point(
            self.state, point, max_dist=24, target_group_id=target_gid,
            exclude_first_point=exclude_first,
        )
        if snapped is not None:
            (sx, sy), sgid, _ = snapped
            hovered = ((sx, sy), sgid)
            if self.renderer._hovered_image_point != hovered:
                self.renderer._hovered_image_point = hovered
                self._redraw()
            return

        if self.renderer._hovered_image_point is not None:
            self._clear_hover()
            self._redraw()

    def _on_left_click(self, event: tk.Event) -> None:
        if self.renderer is None or self.image_bgr is None:
            return
        point = self.renderer.canvas_to_image_xy(event.x, event.y)
        if point is None:
            return

        # ── 新建枝条等待首击（必须在主干线段上）──
        if self._new_branch_waiting_trunk:
            trunk = self.state.groups.get("trunk")
            if trunk is None or len(trunk.points) < 2:
                self.status.config(text="⚠ 主干点数不足")
                return
            snapped = snap_to_existing_line(
                self.state, point, target_group_id="trunk", max_dist=15)
            if snapped is None:
                self.status.config(text="⚠ 新建枝条起点必须在主干线段上（青色高亮区）")
                return
            snapped_point, _ = snapped
            group = self.state.groups[self.state.active_group_id]
            new_idx = group.add_point(snapped_point)
            group.fork_origin_group = "trunk"
            gid = self.state.active_group_id
            self._undo_stack.append({"type": "add_point", "gid": gid, "point_idx": new_idx})
            self._new_branch_waiting_trunk = False
            self._clear_hover()
            self._sync_renderer_flags()
            self._refresh_group_list()
            self._redraw()
            self.status.config(
                text=f"枝条起点: ({snapped_point[0]}, {snapped_point[1]}) (来自主干) — "
                     f"继续点击后续点，hover已有节点可随时分叉"
            )
            self._save_current(silent=True)
            return

        if self.state.active_group_id is None or self.state.active_group_id not in self.state.groups:
            messagebox.showinfo("提示", "请先选择标注模式")
            return

        group = self.state.groups[self.state.active_group_id]

        # ── 枝条组：hover已有节点 = 切换当前分叉起点，不立即连线 ──
        if group.group_type == "branch":
            if self.renderer._hovered_image_point is not None:
                (hx, hy), origin_gid = self.renderer._hovered_image_point
                snapped = snap_to_existing_point(
                    self.state, (hx, hy), max_dist=1, target_group_id=group.group_id,
                    exclude_first_point=True,
                )
                if snapped is not None:
                    _pt, _gid, idx = snapped
                    prev_anchor = group.current_anchor_idx
                    group.current_anchor_idx = idx
                    gid = self.state.active_group_id
                    self._undo_stack.append({
                        "type": "set_anchor",
                        "gid": gid,
                        "prev_anchor_idx": prev_anchor,
                        "new_anchor_idx": idx,
                    })
                    self._clear_hover()
                    self._refresh_group_list()
                    self._redraw()
                    self.status.config(
                        text=f"已选中分叉起点: ({hx}, {hy}) | 当前锚点切换到 '{gid}' 的节点 {idx}"
                    )
                    self._save_current(silent=True)
                    return
                gid = self.state.active_group_id
                self.status.config(text=f"⚠ 当前只能在 '{gid}' 自己的已有节点上继续分叉")
                return

        # ── 常规添加 ──
        gid = self.state.active_group_id
        if group.group_type == "branch":
            prev_anchor = group.current_anchor_idx
            new_idx = group.add_point(point)
            if prev_anchor is not None and prev_anchor != new_idx:
                group.add_edge(prev_anchor, new_idx)
            self._undo_stack.append({
                "type": "add_point",
                "gid": gid,
                "point_idx": new_idx,
                "prev_anchor_idx": prev_anchor,
                "added_edge": (prev_anchor, new_idx) if prev_anchor is not None and prev_anchor != new_idx else None,
            })
        else:
            group.points.append(point)
            group.rebuild_linear_edges()
            self._undo_stack.append({"type": "add_point", "gid": gid, "point_idx": len(group.points) - 1})
        self._refresh_group_list()
        self._redraw()
        self.status.config(
            text=f"已添加点: ({point[0]}, {point[1]}) → "
                 f"'{gid}' ({len(group.points)}点)"
        )
        self._save_current(silent=True)

    def _on_pan_start(self, event: tk.Event) -> None:
        self._is_panning = True
        self._pan_start = (event.x, event.y)

    def _on_pan_drag(self, event: tk.Event) -> None:
        if not self._is_panning or self.renderer is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.renderer.view_offset_x += dx
        self.renderer.view_offset_y += dy
        self._pan_start = (event.x, event.y)
        self._redraw()

    def _on_pan_end(self, event: tk.Event) -> None:
        self._is_panning = False

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if self.renderer is None:
            return
        factor = 1.1 if event.delta > 0 else 0.9
        new_s = self.renderer.view_scale * factor
        if 0.1 <= new_s <= 8.0:
            self.renderer.view_scale = new_s
            self._redraw()

    def _undo_last(self) -> None:
        if not self._undo_stack:
            self.status.config(text="没有可撤销的操作")
            return
        action = self._undo_stack.pop()

        if action["type"] == "add_point":
            gid = action["gid"]
            group = self.state.groups.get(gid)
            if group is None or not group.points:
                self.status.config(text="撤销失败: 分组数据异常")
                return
            point_idx = action.get("point_idx", len(group.points) - 1)
            if group.group_type == "branch":
                added_edge = action.get("added_edge")
                if added_edge is not None and group.edges and group.edges[-1] == added_edge:
                    group.edges.pop()
                if point_idx != len(group.points) - 1:
                    self.status.config(text="撤销失败: 当前仅支持撤销最后一个新增点")
                    return
                removed = group.points.pop()
                group.current_anchor_idx = action.get("prev_anchor_idx")
            else:
                removed = group.points.pop()
                group.rebuild_linear_edges()
            self.state.active_group_id = gid
            self._refresh_group_list()
            self._redraw()
            info = f"已撤销点: ({removed[0]}, {removed[1]})"

            if len(group.points) == 0 and group.group_type == "branch":
                self._cleanup_empty_branch_group(gid)
                info += f" | 同时移除了空枝条组 '{gid}'"
            self.status.config(text=info)
            self._save_current(silent=True)
            return

        if action["type"] == "set_anchor":
            gid = action["gid"]
            group = self.state.groups.get(gid)
            if group is None:
                self.status.config(text="撤销失败: 分组不存在")
                return
            group.current_anchor_idx = action.get("prev_anchor_idx")
            self.state.active_group_id = gid
            self._clear_hover()
            self._refresh_group_list()
            self._redraw()
            self.status.config(text=f"已撤销分叉起点切换: '{gid}'")
            self._save_current(silent=True)
            return

        if action["type"] == "new_branch":
            gid = action["gid"]
            if gid in self.state.groups:
                if len(self.state.groups[gid].points) == 0:
                    del self.state.groups[gid]
                    if self.state.branch_count > 0:
                        self.state.branch_count -= 1
                    if self.state.color_index > 0:
                        self.state.color_index -= 1
            self.state.active_group_id = action["prev_active"]
            self._new_branch_waiting_trunk = action["prev_waiting"]
            self._clear_hover()
            self._sync_renderer_flags()
            self._refresh_group_list()
            self._redraw()
            self.status.config(text=f"已撤销: 新建枝条 '{gid}'")
            self._save_current(silent=True)
            return

        self.status.config(text="未知操作类型, 无法撤销")

    def _cleanup_empty_branch_group(self, gid: str) -> None:
        if gid not in self.state.groups:
            return
        if len(self.state.groups[gid].points) > 0:
            return
        if self.state.groups[gid].group_type != "branch":
            return
        del self.state.groups[gid]
        if self.state.branch_count > 0:
            self.state.branch_count -= 1
        if self.state.color_index > 0:
            self.state.color_index -= 1
        if self.state.active_group_id == gid:
            self.state.active_group_id = "trunk" if "trunk" in self.state.groups else None
        self._new_branch_waiting_trunk = False
        self._clear_hover()
        self._sync_renderer_flags()

    def _on_ctrl_z(self, event: tk.Event) -> None:
        self._undo_last()

    def _on_ctrl_a(self, event: tk.Event) -> None:
        if self._new_branch_waiting_trunk:
            return
        self._mode_new_branch()

    def _delete_group(self) -> None:
        gid = self.state.active_group_id
        if gid is None or gid not in self.state.groups:
            return
        if not messagebox.askyesno("确认删除", f"确定要删除分组 '{gid}' 吗？"):
            return
        del self.state.groups[gid]
        self._undo_stack = [a for a in self._undo_stack
                            if a.get("gid") != gid]
        remaining = list(self.state.groups.keys())
        self.state.active_group_id = remaining[0] if remaining else None
        self._new_branch_waiting_trunk = False
        self._clear_hover()
        self._sync_renderer_flags()
        self._refresh_group_list()
        self._redraw()
        self.status.config(text=f"已删除分组 '{gid}'")
        self._save_current(silent=True)

    def _clear_all(self) -> None:
        if not self.state.groups:
            return
        if not messagebox.askyesno("确认清空", "确定要清空全部标注数据吗？"):
            return
        self.state.clear()
        self._undo_stack.clear()
        self._new_branch_waiting_trunk = False
        self._sync_renderer_flags()
        self._refresh_group_list()
        self._redraw()
        self.status.config(text="已清空全部标注")
        self._save_current(silent=True)

    def _on_group_select(self, event: tk.Event) -> None:
        sel = self.group_listbox.curselection()
        if not sel:
            return
        gid = self.group_listbox.get(sel[0]).split(" ", 1)[0]
        if gid in self.state.groups:
            self.state.active_group_id = gid
            self._new_branch_waiting_trunk = False
            self._clear_hover()
            self._sync_renderer_flags()
            g = self.state.groups[gid]
            self.status.config(
                text=f"已切换到 '{gid}' ({g.group_type}) | {len(g.points)}点"
            )
            self._redraw()

    def _refresh_group_list(self) -> None:
        self.group_listbox.delete(0, tk.END)
        for gid, group in self.state.groups.items():
            n_pts = len(group.points)
            label = f"{gid}  [{group.group_type}]  {n_pts}点"
            self.group_listbox.insert(tk.END, label)
            idx = self.group_listbox.size() - 1
            self.group_listbox.itemconfig(idx, fg=group.color_hex, bg="#1e1e1e")

    def _save_current(self, silent: bool = False) -> None:
        if self._image_folder is None or self._current_index < 0:
            return
        if self.image_bgr is None:
            return
        stem = self._image_files[self._current_index].stem
        save_annotation(
            self._image_folder, stem, self.image_bgr,
            self.height, self.width,
            str(self._image_files[self._current_index]),
            self.state,
            session=self._collect_session(),
        )
        if not silent:
            save_dir = self._image_folder / "skeleton_annotations"
            self.status.config(text=f"已保存到: {save_dir}")
            messagebox.showinfo("保存成功", f"标注已保存到:\n{save_dir}")

    def _collect_session(self) -> dict:
        return {
            "active_group_id": self.state.active_group_id,
            "new_branch_waiting_trunk": self._new_branch_waiting_trunk,
            "undo_stack": self._undo_stack,
        }

    def _restore_session(self, session: dict) -> None:
        self._undo_stack = session.get("undo_stack", []) if isinstance(session, dict) else []
        active_group_id = session.get("active_group_id") if isinstance(session, dict) else None
        self.state.active_group_id = (
            active_group_id if active_group_id in self.state.groups else self.state.active_group_id
        )
        waiting = bool(session.get("new_branch_waiting_trunk")) if isinstance(session, dict) else False
        self._new_branch_waiting_trunk = waiting and self.state.active_group_id in self.state.groups
        self._clear_hover()
        self._sync_renderer_flags()

    def _manual_save(self) -> None:
        self._save_current(silent=False)

    def _redraw(self) -> None:
        if self.renderer is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                text="请点击「打开图片文件夹」加载图片", fill="#888888",
                font=("Microsoft YaHei", 14),
            )
            return
        self.renderer.show_mask = self._show_mask.get()
        self.renderer.branch_mask = self.branch_mask
        self.renderer.redraw(self.state)


def main() -> None:
    root = tk.Tk()
    app = SkeletonAnnotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
