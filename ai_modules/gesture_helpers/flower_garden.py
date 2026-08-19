import math
import random
import time
import cv2
import numpy as np

# Cute Cartoon & Vibrant Color Palettes (BGR format)
BUTTERFLY_PALETTES = [
    # Lavender Pink
    {'outer': (180, 130, 255), 'inner': (220, 180, 255), 'dots': (255, 255, 255), 'body': (100, 50, 120), 'outline': (80, 30, 100)},
    # Sky Cyan
    {'outer': (255, 190, 80), 'inner': (255, 230, 160), 'dots': (255, 255, 255), 'body': (130, 70, 30), 'outline': (100, 40, 10)},
    # Golden Sunny
    {'outer': (80, 215, 255), 'inner': (160, 240, 255), 'dots': (255, 255, 255), 'body': (40, 90, 160), 'outline': (20, 50, 110)},
    # Magenta Purple
    {'outer': (220, 50, 255), 'inner': (240, 150, 255), 'dots': (255, 255, 255), 'body': (100, 20, 110), 'outline': (60, 0, 80)},
    # Emerald Mint
    {'outer': (150, 245, 120), 'inner': (200, 255, 180), 'dots': (255, 255, 255), 'body': (40, 110, 50), 'outline': (20, 70, 30)}
]

FLOWER_PALETTES = {
    "Cherry Blossom 🌸": {'petals': (203, 192, 255), 'inner': (147, 20, 255), 'center': (50, 0, 150), 'stem': (80, 180, 80)},
    "Golden Lotus 🌻": {'petals': (50, 215, 255), 'inner': (0, 165, 255), 'center': (0, 80, 200), 'stem': (50, 150, 50)},
    "Cyber Violet 🪻": {'petals': (255, 100, 200), 'inner': (200, 50, 150), 'center': (255, 255, 0), 'stem': (100, 200, 100)},
    "Celestial Azure 🩵": {'petals': (255, 220, 100), 'inner': (255, 180, 50), 'center': (150, 50, 0), 'stem': (60, 160, 60)},
    "Royal Rose 🌹": {'petals': (100, 50, 255), 'inner': (50, 0, 200), 'center': (0, 215, 255), 'stem': (40, 140, 40)}
}


class Butterfly:
    """Animated butterfly particle with flapping wings and floating flight physics."""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        angle = random.uniform(-math.pi * 0.85, -math.pi * 0.15)
        speed = random.uniform(2.0, 4.2)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

        self.size = random.uniform(24.0, 36.0)
        self.palette = random.choice(BUTTERFLY_PALETTES)

        self.wing_phase = random.uniform(0, math.pi * 2)
        self.flap_speed = random.uniform(10.0, 16.0)

        self.life = 1.0
        self.decay = random.uniform(0.004, 0.009)
        self.age = 0.0

    def update(self):
        self.age += 0.05
        self.vx += math.sin(self.age * 2.2) * 0.35
        self.vy += math.sin(self.age * 1.8) * 0.18 - 0.04

        self.x += self.vx
        self.y += self.vy

        self.vx *= 0.97
        self.vy *= 0.97

        self.life -= self.decay
        return self.life > 0

    def draw(self, frame):
        if self.life <= 0:
            return

        alpha = max(0.0, min(1.0, self.life))
        flap = abs(math.sin(self.age * self.flap_speed + self.wing_phase))
        flap = max(0.18, flap)

        wing_w = int(self.size * flap)
        wing_h = int(self.size * 0.9)

        cx, cy = int(self.x), int(self.y)
        h_frame, w_frame, _ = frame.shape
        if not (-60 <= cx <= w_frame + 60 and -60 <= cy <= h_frame + 60):
            return

        flight_angle = math.atan2(self.vy, self.vx) + math.pi / 2
        rot_deg = math.degrees(flight_angle) * 0.3

        c_outer = tuple(int(c * alpha) for c in self.palette['outer'])
        c_inner = tuple(int(c * alpha) for c in self.palette['inner'])
        c_dots = tuple(int(c * alpha) for c in self.palette['dots'])
        c_body = tuple(int(c * alpha) for c in self.palette['body'])
        c_outline = tuple(int(c * alpha) for c in self.palette['outline'])

        # Upper wings
        r_upper_x = max(3, int(wing_w * 0.85))
        r_upper_y = max(4, int(wing_h * 0.95))

        ul_x = int(cx - wing_w * 0.65)
        ul_y = int(cy - wing_h * 0.35)
        ur_x = int(cx + wing_w * 0.65)
        ur_y = int(cy - wing_h * 0.35)

        if wing_w > 5:
            cv2.ellipse(frame, (ul_x, ul_y), (r_upper_x + 2, r_upper_y + 2), int(rot_deg - 20), 0, 360, c_outline, -1, cv2.LINE_AA)
            cv2.ellipse(frame, (ur_x, ur_y), (r_upper_x + 2, r_upper_y + 2), int(rot_deg + 20), 0, 360, c_outline, -1, cv2.LINE_AA)

        cv2.ellipse(frame, (ul_x, ul_y), (r_upper_x, r_upper_y), int(rot_deg - 20), 0, 360, c_outer, -1, cv2.LINE_AA)
        cv2.ellipse(frame, (ur_x, ur_y), (r_upper_x, r_upper_y), int(rot_deg + 20), 0, 360, c_outer, -1, cv2.LINE_AA)

        if wing_w > 8:
            r_in_x = max(2, int(r_upper_x * 0.65))
            r_in_y = max(2, int(r_upper_y * 0.65))
            cv2.ellipse(frame, (ul_x + 2, ul_y + 2), (r_in_x, r_in_y), int(rot_deg - 20), 0, 360, c_inner, -1, cv2.LINE_AA)
            cv2.ellipse(frame, (ur_x - 2, ur_y + 2), (r_in_x, r_in_y), int(rot_deg + 20), 0, 360, c_inner, -1, cv2.LINE_AA)

            dot_r = max(2, int(4 * flap * alpha))
            if dot_r >= 2:
                cv2.circle(frame, (ul_x - int(r_upper_x * 0.3), ul_y - int(r_upper_y * 0.3)), dot_r, c_dots, -1, cv2.LINE_AA)
                cv2.circle(frame, (ur_x + int(r_upper_x * 0.3), ur_y - int(r_upper_y * 0.3)), dot_r, c_dots, -1, cv2.LINE_AA)

        # Lower wings
        r_lower_x = max(2, int(wing_w * 0.65))
        r_lower_y = max(3, int(wing_h * 0.7))

        ll_x = int(cx - wing_w * 0.45)
        ll_y = int(cy + wing_h * 0.4)
        lr_x = int(cx + wing_w * 0.45)
        lr_y = int(cy + wing_h * 0.4)

        if wing_w > 5:
            cv2.ellipse(frame, (ll_x, ll_y), (r_lower_x + 2, r_lower_y + 2), int(rot_deg + 15), 0, 360, c_outline, -1, cv2.LINE_AA)
            cv2.ellipse(frame, (lr_x, lr_y), (r_lower_x + 2, r_lower_y + 2), int(rot_deg - 15), 0, 360, c_outline, -1, cv2.LINE_AA)

        cv2.ellipse(frame, (ll_x, ll_y), (r_lower_x, r_lower_y), int(rot_deg + 15), 0, 360, c_outer, -1, cv2.LINE_AA)
        cv2.ellipse(frame, (lr_x, lr_y), (r_lower_x, r_lower_y), int(rot_deg - 15), 0, 360, c_outer, -1, cv2.LINE_AA)

        # Body
        body_h = max(6, int(self.size * 0.5))
        body_w = max(3, int(6 * alpha))
        cv2.ellipse(frame, (cx, cy + 2), (body_w + 1, body_h + 1), int(rot_deg), 0, 360, c_outline, -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy + 2), (body_w, body_h), int(rot_deg), 0, 360, c_body, -1, cv2.LINE_AA)

        # Head & Antennae
        head_r = max(3, int(4 * alpha))
        head_y = cy - body_h // 2 - 2
        cv2.circle(frame, (cx, head_y), head_r + 1, c_outline, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, head_y), head_r, c_body, -1, cv2.LINE_AA)

        ant_len = max(8, int(self.size * 0.45))
        ant_left_x = int(cx - ant_len * 0.7)
        ant_right_x = int(cx + ant_len * 0.7)
        ant_tip_y = head_y - ant_len

        cv2.line(frame, (cx, head_y), (ant_left_x, ant_tip_y), c_outline, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, head_y), (ant_right_x, ant_tip_y), c_outline, 2, cv2.LINE_AA)

        bulb_r = max(2, int(3 * alpha))
        cv2.circle(frame, (ant_left_x, ant_tip_y), bulb_r, c_outer, -1, cv2.LINE_AA)
        cv2.circle(frame, (ant_right_x, ant_tip_y), bulb_r, c_outer, -1, cv2.LINE_AA)


class Flower:
    """Rich multi-layered blooming flower with stem, leaves, and swaying animation."""
    def __init__(self, x, y, flower_style="Cherry Blossom 🌸"):
        self.x = float(x)
        self.y = float(y)
        
        # Select or match palette
        if flower_style in FLOWER_PALETTES:
            self.palette = FLOWER_PALETTES[flower_style]
        else:
            self.palette = random.choice(list(FLOWER_PALETTES.values()))

        self.max_radius = random.uniform(28.0, 42.0)
        self.num_petals = random.choice([5, 6, 8, 10, 12])
        self.growth = 0.0
        self.growth_speed = random.uniform(0.04, 0.07)

        self.sway_phase = random.uniform(0, math.pi * 2)
        self.sway_speed = random.uniform(1.2, 2.5)

        self.created_time = time.time()
        self.stem_height = random.uniform(65.0, 110.0)

    def update(self):
        if self.growth < 1.0:
            self.growth = min(1.0, self.growth + self.growth_speed)

    def draw(self, frame, current_time):
        cx = int(self.x)
        cy = int(self.y)

        sway = math.sin((current_time - self.created_time) * self.sway_speed + self.sway_phase) * 6.0 * self.growth
        head_x = int(cx + sway)
        head_y = cy

        # Stem & Leaves
        stem_bottom_y = min(frame.shape[0], int(cy + self.stem_height * self.growth))
        stem_color = self.palette['stem']

        curve_x = int(cx + sway * 0.5)
        curve_y = int((head_y + stem_bottom_y) / 2)
        pts_stem = np.array([[head_x, head_y], [curve_x, curve_y], [cx, stem_bottom_y]], np.int32)
        cv2.polylines(frame, [pts_stem], False, stem_color, max(2, int(4 * self.growth)), cv2.LINE_AA)

        if self.growth > 0.4:
            leaf_scale = (self.growth - 0.4) / 0.6
            leaf_w = int(18 * leaf_scale)
            cv2.ellipse(frame, (curve_x - leaf_w // 2, curve_y), (leaf_w, 6), -25, 0, 360, stem_color, -1, cv2.LINE_AA)
            cv2.ellipse(frame, (curve_x + leaf_w // 2, curve_y + 8), (leaf_w, 6), 25, 0, 360, stem_color, -1, cv2.LINE_AA)

        # Petals
        if self.growth > 0.15:
            current_radius = self.max_radius * ((self.growth - 0.15) / 0.85)
            petals_col = self.palette['petals']
            inner_col = self.palette['inner']
            center_col = self.palette['center']

            # Outer Petal layer
            for i in range(self.num_petals):
                angle = (2 * math.pi / self.num_petals) * i + sway * 0.05
                px = head_x + math.cos(angle) * current_radius * 0.6
                py = head_y + math.sin(angle) * current_radius * 0.6

                rx = max(4, int(current_radius * 0.45))
                ry = max(6, int(current_radius * 0.75))
                rot_deg = math.degrees(angle) + 90

                cv2.ellipse(frame, (int(px), int(py)), (rx, ry), rot_deg, 0, 360, petals_col, -1, cv2.LINE_AA)
                cv2.ellipse(frame, (int(px), int(py)), (rx, ry), rot_deg, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)

            # Inner depth Petal layer
            if current_radius > 12:
                for i in range(self.num_petals):
                    angle = (2 * math.pi / self.num_petals) * i + (math.pi / self.num_petals) + sway * 0.05
                    px = head_x + math.cos(angle) * current_radius * 0.35
                    py = head_y + math.sin(angle) * current_radius * 0.35
                    rx = max(2, int(current_radius * 0.3))
                    ry = max(4, int(current_radius * 0.5))
                    rot_deg = math.degrees(angle) + 90
                    cv2.ellipse(frame, (int(px), int(py)), (rx, ry), rot_deg, 0, 360, inner_col, -1, cv2.LINE_AA)

            # Flower Center Core
            core_r = max(4, int(current_radius * 0.3))
            cv2.circle(frame, (head_x, head_y), core_r, center_col, -1, cv2.LINE_AA)
            cv2.circle(frame, (head_x, head_y), core_r, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (head_x, head_y), max(2, core_r // 2), (255, 255, 200), -1, cv2.LINE_AA)


class SparkleParticle:
    """Sparkle particle for flowers and butterfly flight trails."""
    def __init__(self, x, y, color=(255, 255, 255)):
        self.x = float(x)
        self.y = float(y)
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(0.5, 3.0)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 0.5
        self.color = color
        self.size = random.uniform(3.0, 7.0)
        self.life = 1.0
        self.decay = random.uniform(0.03, 0.08)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        self.size = max(0.5, self.size * 0.94)
        return self.life > 0

    def draw(self, frame):
        if self.life <= 0:
            return
        cx, cy = int(self.x), int(self.y)
        h, w, _ = frame.shape
        if 0 <= cx < w and 0 <= cy < h:
            r = int(self.size)
            alpha_col = tuple(int(c * self.life) for c in self.color)
            cv2.circle(frame, (cx, cy), r, alpha_col, -1, cv2.LINE_AA)
            if r > 2:
                cv2.line(frame, (cx - r - 2, cy), (cx + r + 2, cy), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(frame, (cx, cy - r - 2), (cx, cy + r + 2), (255, 255, 255), 1, cv2.LINE_AA)


class FlowerGardenEngine:
    """Manages persistent garden flowers, animated butterflies, and gesture interactions."""
    def __init__(self):
        self.flowers = []
        self.butterflies = []
        self.sparkles = []
        self.last_butterfly_time = 0
        self.is_pinching_by_hand = {}

    def process_hand_gestures(self, hands_data, flower_style="Cherry Blossom 🌸"):
        now = time.time()
        active_hands = set()

        for hand in hands_data:
            handedness = hand.get('handedness', 'Right')
            active_hands.add(handedness)
            gesture = hand['gesture']

            is_currently_pinching = (gesture == 'PINCH')
            was_pinching = self.is_pinching_by_hand.get(handedness, False)

            # 1. PINCH (👌/🤏): Trigger EXACTLY ONE flower per pinch gesture
            if is_currently_pinching and not was_pinching:
                pos = hand['pinch_point']
                self.flowers.append(Flower(pos[0], pos[1], flower_style))
                # Add sparkle explosion
                for _ in range(8):
                    self.sparkles.append(SparkleParticle(pos[0], pos[1], (255, 220, 150)))

            # Update pinch state for this hand
            self.is_pinching_by_hand[handedness] = is_currently_pinching

            # 2. PEACE (✌️): Spawn Flying Butterflies!
            if gesture == 'PEACE':
                pos = hand['landmarks_px'][8]  # Index tip
                if now - self.last_butterfly_time > 0.18:
                    for _ in range(2):
                        if len(self.butterflies) < 60:
                            self.butterflies.append(Butterfly(pos[0], pos[1]))
                    self.last_butterfly_time = now
                    for _ in range(4):
                        self.sparkles.append(SparkleParticle(pos[0], pos[1], (180, 240, 255)))

            # 3. ERASE (🖐️ Open Palm): Erase Flowers & Butterflies under wide open palm
            elif gesture == 'OPEN_PALM':
                eraser_pos = hand['palm_center']
                erase_radius = 80  # pixels
                # Filter flowers
                self.flowers = [
                    f for f in self.flowers
                    if math.hypot(f.x - eraser_pos[0], f.y - eraser_pos[1]) > erase_radius
                ]
                # Filter butterflies
                self.butterflies = [
                    b for b in self.butterflies
                    if math.hypot(b.x - eraser_pos[0], b.y - eraser_pos[1]) > erase_radius
                ]

        # Reset pinch state for hands no longer present in frame
        for h in list(self.is_pinching_by_hand.keys()):
            if h not in active_hands:
                self.is_pinching_by_hand[h] = False

    def draw(self, frame, hands_data=None):
        current_time = time.time()

        # Update and draw flowers
        for flower in self.flowers:
            flower.update()
            flower.draw(frame, current_time)

        # Update and draw butterflies
        self.butterflies = [b for b in self.butterflies if b.update()]
        for b in self.butterflies:
            b.draw(frame)

        # Update and draw sparkles
        self.sparkles = [s for s in self.sparkles if s.update()]
        for s in self.sparkles:
            s.draw(frame)

        # Draw Eraser Glow for OPEN_PALM erase gesture
        if hands_data:
            for hand in hands_data:
                if hand['gesture'] == 'OPEN_PALM':
                    cx, cy = hand['palm_center']
                    overlay = frame.copy()
                    color = (255, 120, 50)
                    text = "🖐️ ERASE GESTURE"

                    cv2.circle(overlay, (cx, cy), 80, color, -1, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.circle(frame, (cx, cy), 80, color, 2, cv2.LINE_AA)

                    cv2.putText(frame, text, (cx - 65, cy - 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, text, (cx - 65, cy - 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Instruction overlay legend at bottom of video
        h, w, _ = frame.shape
        legend_text = f"Flowers: {len(self.flowers)}  Butterflies: {len(self.butterflies)}  |  🤏 Pinch: Flower  |  ✌️ Peace: Butterfly  |  🖐️ Open Palm: Erase"
        cv2.putText(frame, legend_text, (15, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, legend_text, (15, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    def clear(self):
        """Remove all flowers, butterflies, and sparkles."""
        self.flowers.clear()
        self.butterflies.clear()
        self.sparkles.clear()
        self.is_pinching_by_hand.clear()
