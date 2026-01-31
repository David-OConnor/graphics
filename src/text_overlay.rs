//! For drawing text on over 3D graphics, using EGUI's painer.

use egui::{Align2, Color32, FontFamily, FontId, Pos2};
use lin_alg::f32::Vec3;

use crate::{UiSettings, graphics::GraphicsState, gui::GuiState};

#[derive(Debug, Clone)]
pub struct TextOverlay {
    pub text: String,
    pub size: f32,
    /// Red, Greed, Blue, alpha
    pub color: (u8, u8, u8, u8),
    pub font_family: FontFamily,
}

impl Default for TextOverlay {
    fn default() -> Self {
        Self {
            text: String::new(),
            size: 13.,
            color: (255, 255, 255, 255),
            font_family: FontFamily::Proportional,
        }
    }
}

pub(crate) fn draw_text_overlay(
    graphics_state: &GraphicsState,
    gui: &GuiState,
    ui_settings: &UiSettings,
    // These are in physical pixels.
    width: u32,
    height: u32,
) {
    let ctx = gui.egui_state.egui_ctx();

    // Compute label positions in screen space
    let labels = graphics_state.collect_entity_labels(
        width,
        height,
        ui_settings,
        gui.size,
        ctx.pixels_per_point(),
    );

    // Paint in the foreground layer
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("entity_labels"),
    ));
    for (pos, overlay) in labels {
        let (r, g, b, a) = overlay.color;

        painter.text(
            pos,
            Align2::CENTER_BOTTOM,
            &overlay.text,
            // todo: Font size may need to be part of the label.
            FontId::new(overlay.size, overlay.font_family.clone()),
            Color32::from_rgba_unmultiplied(r, g, b, a),
        );
    }
}

impl GraphicsState {
    /// We use this for the text overlay.
    /// Project a world-space point to screen-space (in egui points).
    /// Returns None if behind camera or outside clip space.
    pub fn world_to_screen(
        &self,
        world: Vec3,
        // In physical pixels
        width: u32,
        height: u32,
        ui_settings: &UiSettings,
        ui_size: (f32, f32),
        pixels_per_pt: f32,
    ) -> Option<Pos2> {
        let (vx, vy, vw, vh) =
            self.viewport_rect(ui_size, width, height, ui_settings, pixels_per_pt);

        let (in_view, ndc) = self.scene.camera.in_view(world);
        if !in_view {
            return None;
        }

        // NDC -> pixels in 3D viewport
        let sx = vx + (ndc.0 * 0.5 + 0.5) * vw;
        let sy = vy + (1.0 - (ndc.1 * 0.5 + 0.5)) * vh; // flip Y for top-left origin

        Some(Pos2::new(sx, sy))
    }

    /// Convenience: gather label screen positions for all entities that have `overlay_text`.
    pub fn collect_entity_labels(
        &self,
        // Physical pixels
        width: u32,
        height: u32,
        ui_settings: &UiSettings,
        ui_size: (f32, f32),
        pixels_per_pt: f32,
    ) -> Vec<(Pos2, &TextOverlay)> {
        let mut out = Vec::new();

        for e in &self.scene.entities {
            if let Some(overlay) = &e.overlay_text {
                // Slight vertical offset above the entity (tune as you like).
                let label_world = Vec3 {
                    x: e.position.x,
                    y: e.position.y + 0.05 * e.scale, // small lift
                    z: e.position.z,
                };
                if let Some(p) = self.world_to_screen(
                    label_world,
                    width,
                    height,
                    ui_settings,
                    ui_size,
                    pixels_per_pt,
                ) {
                    out.push((p, overlay));
                }
            }
        }
        out
    }
}
