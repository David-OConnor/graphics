//! For drawing text on over 3D graphics, using EGUI's painer.

use egui::{Align2, Color32, FontFamily, FontId, Pos2};
use lin_alg::f32::{Vec3, Vec4};

use crate::{
    UiSettings, graphics::GraphicsState, gui::GuiState, types::FramerateDisplay, viewport_rect,
};

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

    if labels.is_empty() {
        return;
    }

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

/// Draw the frame rate readout in the corner of the 3D display area selected by
/// `GraphicsSettings::display_framerate`.
pub(crate) fn draw_framerate(
    graphics_state: &GraphicsState,
    gui: &GuiState,
    ui_settings: &UiSettings,
    // These are in physical pixels.
    width: u32,
    height: u32,
) {
    // Skip drawing until the first measurement window has completed.
    if graphics_state.framerate_display == FramerateDisplay::Disabled
        || graphics_state.fps_value <= 0.
    {
        return;
    }

    let ctx = gui.egui_state.egui_ctx();
    let pixels_per_pt = ctx.pixels_per_point();

    // Convert physical pixels to logical pixels (egui points), as with `world_to_screen`.
    let logical_width = (width as f32 / pixels_per_pt).round() as u32;
    let logical_height = (height as f32 / pixels_per_pt).round() as u32;

    let (x, y, eff_width, eff_height) = viewport_rect(
        gui.size,
        logical_width,
        logical_height,
        ui_settings,
        pixels_per_pt,
    );

    const MARGIN: f32 = 10.;
    let (pos, align) = match graphics_state.framerate_display {
        FramerateDisplay::TopLeft => (Pos2::new(x + MARGIN, y + MARGIN), Align2::LEFT_TOP),
        FramerateDisplay::TopRight => (
            Pos2::new(x + eff_width - MARGIN, y + MARGIN),
            Align2::RIGHT_TOP,
        ),
        FramerateDisplay::BottomLeft => (
            Pos2::new(x + MARGIN, y + eff_height - MARGIN),
            Align2::LEFT_BOTTOM,
        ),
        FramerateDisplay::BottomRight => (
            Pos2::new(x + eff_width - MARGIN, y + eff_height - MARGIN),
            Align2::RIGHT_BOTTOM,
        ),
        FramerateDisplay::Disabled => unreachable!(),
    };

    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("framerate_display"),
    ));

    let text = format!("{:.0} FPS", graphics_state.fps_value);
    let font = FontId::new(14., FontFamily::Monospace);

    // A dark shadow behind the text, so it's legible over light backgrounds.
    painter.text(
        pos + egui::vec2(1., 1.),
        align,
        &text,
        font.clone(),
        Color32::from_rgba_unmultiplied(0, 0, 0, 180),
    );
    painter.text(pos, align, &text, font, Color32::WHITE);
}

impl GraphicsState {
    /// Project a world-space point to screen-space (in egui points).
    /// Returns None if behind camera or outside clip space.
    /// Note: `collect_entity_labels` inlines this math so the viewport and projection
    /// matrix are computed once per frame rather than once per label.
    #[allow(dead_code)]
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
        // Convert physical pixels to logical pixels (egui points) so the result is
        // in egui point space and Pos2 is placed correctly on HiDPI displays.
        let logical_width = (width as f32 / pixels_per_pt).round() as u32;
        let logical_height = (height as f32 / pixels_per_pt).round() as u32;

        let (x, y, eff_width, eff_height) = viewport_rect(
            ui_size,
            logical_width,
            logical_height,
            ui_settings,
            pixels_per_pt,
        );

        let (in_view, ndc) = self.scene.camera.in_view(world);
        if !in_view {
            return None;
        }

        // NDC -> pixels in 3D viewport
        let sx = x + (ndc.0 * 0.5 + 0.5) * eff_width;
        let sy = y + (1.0 - (ndc.1 * 0.5 + 0.5)) * eff_height; // flip Y for top-left origin

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

        if !self.scene.entities.iter().any(|e| e.overlay_text.is_some()) {
            return out;
        }

        // The viewport rect and projection matrix are shared by all labels; compute
        // them once here instead of once per label, as `world_to_screen` would.
        let logical_width = (width as f32 / pixels_per_pt).round() as u32;
        let logical_height = (height as f32 / pixels_per_pt).round() as u32;

        let (x, y, eff_width, eff_height) = viewport_rect(
            ui_size,
            logical_width,
            logical_height,
            ui_settings,
            pixels_per_pt,
        );

        let proj_view = self.scene.camera.proj_mat.clone() * self.scene.camera.view_mat();

        for e in &self.scene.entities {
            if let Some(overlay) = &e.overlay_text {
                // Slight vertical offset above the entity (tune as you like).
                let label_world = Vec3 {
                    x: e.position.x,
                    y: e.position.y + 0.05 * e.scale, // small lift
                    z: e.position.z,
                };

                // Same math as `Camera::in_view`, using the precomputed matrix.
                let p4 =
                    proj_view.clone() * Vec4::new(label_world.x, label_world.y, label_world.z, 1.0);
                if p4.w <= 0.0 {
                    continue;
                }

                let inv_w = 1.0 / p4.w;
                let ndc_x = p4.x * inv_w;
                let ndc_y = p4.y * inv_w;
                let ndc_z = p4.z * inv_w;

                if !((-1.0..=1.0).contains(&ndc_x)
                    && (-1.0..=1.0).contains(&ndc_y)
                    && (-1.0..=1.0).contains(&ndc_z))
                {
                    continue;
                }

                // NDC -> pixels in 3D viewport
                let sx = x + (ndc_x * 0.5 + 0.5) * eff_width;
                let sy = y + (1.0 - (ndc_y * 0.5 + 0.5)) * eff_height; // flip Y for top-left origin

                out.push((Pos2::new(sx, sy), overlay));
            }
        }
        out
    }
}
