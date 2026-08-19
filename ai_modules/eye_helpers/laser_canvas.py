import math
import random
import time
import cv2
import numpy as np


class LaserParticle:
    """Particle effect for Neon Plasma mode."""

    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-2.0, 2.0)
        self.vy = random.uniform(-2.0, 2.0)
        self.size = random.uniform(3.0, 7.0)
        self.color = color
        self.life = 1.0
        self.decay = random.uniform(0.05, 0.09)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.size = max(0.5, self.size - 0.15)
        self.life -= self.decay

    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = float(self.life)
        radius = max(1, int(self.size))
        color_with_alpha = (
            int(self.color[0] * alpha),
            int(self.color[1] * alpha),
            int(self.color[2] * alpha)
        )
        cv2.circle(frame, (int(self.x), int(self.y)), radius, color_with_alpha, -1, cv2.LINE_AA)


class TargetGame:
    """Gaze Target Lock-on Mini-Game."""

    def __init__(self):
        self.active = False
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.streak = 0
        self.target_x = 320
        self.target_y = 240
        self.target_radius = 45
        self.lock_progress = 0.0  # 0.0 to 1.0
        self.lock_duration = 0.45  # seconds of continuous gaze lock needed
        self.lock_start_time = None
        self.hit_effect_timer = 0
        self.last_hit_pos = (0, 0)

    def reset(self, frame_w=640, frame_h=480):
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.streak = 0
        self.spawn_new_target(frame_w, frame_h)

    def spawn_new_target(self, frame_w, frame_h):
        margin = 90
        w = max(200, frame_w)
        h = max(200, frame_h)
        self.target_x = random.randint(margin, w - margin)
        self.target_y = random.randint(margin, h - margin)
        self.lock_progress = 0.0
        self.lock_start_time = None

    def update(self, gaze_px, is_blink, frame_w, frame_h):
        gx, gy = gaze_px
        dist = math.hypot(gx - self.target_x, gy - self.target_y)

        # Check if gaze is inside target circle
        if dist <= self.target_radius:
            if self.lock_start_time is None:
                self.lock_start_time = time.time()
            elapsed = time.time() - self.lock_start_time
            self.lock_progress = min(1.0, elapsed / self.lock_duration)

            # Hit condition: fully charged lock or blink while inside target
            if self.lock_progress >= 1.0 or is_blink:
                self.hits += 1
                self.streak += 1
                combo_multiplier = min(5, 1 + self.streak // 3)
                self.score += 100 * combo_multiplier

                self.last_hit_pos = (self.target_x, self.target_y)
                self.hit_effect_timer = 15
                self.spawn_new_target(frame_w, frame_h)
                return True
        else:
            self.lock_start_time = None
            self.lock_progress = max(0.0, self.lock_progress - 0.05)

        return False

    def draw(self, frame):
        # Draw Hit Effect Flash
        if self.hit_effect_timer > 0:
            hx, hy = self.last_hit_pos
            r = int(50 + (15 - self.hit_effect_timer) * 4)
            cv2.circle(frame, (hx, hy), r, (166, 227, 161), 3, cv2.LINE_AA)
            cv2.putText(frame, f"+100 HIT! 🔥", (hx - 40, hy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (166, 227, 161), 2, cv2.LINE_AA)
            self.hit_effect_timer -= 1

        tx, ty = self.target_x, self.target_y
        r = self.target_radius

        # Outer Pulsing Target Ring
        pulse_r = int(r + math.sin(time.time() * 6.0) * 4)
        cv2.circle(frame, (tx, ty), pulse_r, (243, 139, 168), 2, cv2.LINE_AA)

        # Crosshair lines
        cv2.line(frame, (tx - r - 10, ty), (tx - r + 5, ty), (243, 139, 168), 2, cv2.LINE_AA)
        cv2.line(frame, (tx + r - 5, ty), (tx + r + 10, ty), (243, 139, 168), 2, cv2.LINE_AA)
        cv2.line(frame, (tx, ty - r - 10), (tx, ty - r + 5), (243, 139, 168), 2, cv2.LINE_AA)
        cv2.line(frame, (tx, ty + r - 5), (tx, ty + r + 10), (243, 139, 168), 2, cv2.LINE_AA)

        # Inner Lock Progress Arc
        if self.lock_progress > 0:
            angle = int(self.lock_progress * 360)
            cv2.ellipse(frame, (tx, ty), (r - 6, r - 6), 0, -90, -90 + angle, (166, 227, 161), -1)

        cv2.circle(frame, (tx, ty), 6, (255, 255, 255), -1, cv2.LINE_AA)

        # In-game HUD Score Banner (Top Left)
        hud_str = f"SCORE: {self.score} | STREAK: {self.streak}x"
        cv2.putText(frame, hud_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, hud_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 179, 135), 2, cv2.LINE_AA)


class LaserCanvasEngine:
    """Renders virtual laser pointers (Cyber Red & Neon Blue) and Gaze Target Mini-Game."""

    def __init__(self):
        self.laser_style = "🔴 Cyber Red Laser"
        self.trail_history = []
        self.particles = []
        self.game = TargetGame()

    def set_laser_style(self, style):
        self.laser_style = style

    def clear(self):
        self.trail_history.clear()
        self.particles.clear()
        self.game.reset()

    def update_and_render(self, frame_bgr, eye_data, visual_mode):
        """Main render pass called every frame."""
        h, w, _ = frame_bgr.shape

        if eye_data is None or 'laser_px' not in eye_data:
            return

        lx, ly = eye_data['laser_px']
        is_blink = eye_data.get('is_blink_event', False)

        # Accumulate Motion Trail History
        now = time.time()
        self.trail_history.append((lx, ly, now))
        self.trail_history = [(x, y, t) for (x, y, t) in self.trail_history if now - t <= 1.0]

        # Handle Target Mini-Game Mode
        if "Game" in visual_mode or "Target" in visual_mode:
            self.game.update((lx, ly), is_blink, w, h)
            self.game.draw(frame_bgr)

        # Render Active Laser Pointer (Cyber Red or Neon Blue)
        self._render_laser_pointer(frame_bgr, lx, ly)

        # Render Particle trail updates
        for p in self.particles[:]:
            p.update()
            p.draw(frame_bgr)
            if p.life <= 0:
                self.particles.remove(p)

    def _render_laser_pointer(self, frame_bgr, lx, ly):
        overlay = frame_bgr.copy()
        is_neon = "Neon" in self.laser_style or "Blue" in self.laser_style

        laser_color = (255, 200, 50) if is_neon else (0, 0, 255)

        # Render Motion Laser Trail
        if len(self.trail_history) > 1:
            for i in range(1, len(self.trail_history)):
                pt1 = self.trail_history[i - 1][:2]
                pt2 = self.trail_history[i][:2]
                alpha = i / float(len(self.trail_history))
                thick = max(1, int(alpha * 5))
                cv2.line(frame_bgr, pt1, pt2, laser_color, thick, cv2.LINE_AA)

        if is_neon:
            # ⚡ Neon Plasma Blue Laser Pointer
            cv2.circle(overlay, (lx, ly), 22, (255, 180, 50), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)

            # Electric crosshair ring
            angle = time.time() * 8.0
            r = 24
            x1 = int(lx + r * math.cos(angle))
            y1 = int(ly + r * math.sin(angle))
            cv2.line(frame_bgr, (lx, ly), (x1, y1), (255, 240, 150), 2, cv2.LINE_AA)
            cv2.circle(frame_bgr, (lx, ly), 6, (255, 255, 255), -1, cv2.LINE_AA)

            if random.random() < 0.6:
                self.particles.append(LaserParticle(lx, ly, (255, 200, 80)))
        else:
            # 🔴 Cyber Red Laser Pointer
            cv2.circle(overlay, (lx, ly), 18, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (lx, ly), 30, (0, 100, 255), 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.4, frame_bgr, 0.6, 0, frame_bgr)

            # Core White Bright Spot
            cv2.circle(frame_bgr, (lx, ly), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (lx, ly), 3, (150, 150, 255), -1, cv2.LINE_AA)
