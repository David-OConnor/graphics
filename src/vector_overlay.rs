//! View-only vector graphics drawn over the 3D viewport with egui's painter.
//!
//! Coordinates inside an overlay are logical pixels (egui points) relative to its anchor. This
//! keeps small HUD elements crisp and independent of the 3D render resolution while still letting
//! callers anchor them to a viewport corner or a world-space position.

use egui::{
    Align2, Color32, Context, FontFamily, FontId, LayerId, Order, Pos2, Rect, Shape, Stroke, Vec2,
};
use lin_alg::f32::Vec3;

use crate::{UiSettings, graphics::GraphicsState, viewport_rect};

pub type OverlayColor = (u8, u8, u8, u8);
pub type OverlayPoint = (f32, f32);

/// Where an overlay's local origin is placed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverlayAnchor {
    ViewportTopLeft,
    ViewportTopRight,
    ViewportBottomLeft,
    ViewportBottomRight,
    ViewportCenter,
    /// Logical pixels from the viewport's top-left corner.
    ViewportPoint(OverlayPoint),
    /// A projected 3D position. The overlay is hidden when the point is outside the camera frustum.
    World(Vec3),
}

/// A stroke used by one of the vector primitives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayStroke {
    pub width: f32,
    pub color: OverlayColor,
}

impl OverlayStroke {
    pub const fn new(width: f32, color: OverlayColor) -> Self {
        Self { width, color }
    }
}

/// Basic, non-interactive shapes for composing a [`VectorOverlay`].
#[derive(Clone, Debug)]
pub enum OverlayPrimitive {
    Line {
        start: OverlayPoint,
        end: OverlayPoint,
        stroke: OverlayStroke,
    },
    Polyline {
        points: Vec<OverlayPoint>,
        stroke: OverlayStroke,
        closed: bool,
    },
    Circle {
        center: OverlayPoint,
        radius: f32,
        fill: Option<OverlayColor>,
        stroke: Option<OverlayStroke>,
    },
    /// Convex polygons cover arrowheads and other compact filled shapes without imposing a
    /// heavyweight path API on callers.
    Polygon {
        points: Vec<OverlayPoint>,
        fill: OverlayColor,
        stroke: Option<OverlayStroke>,
    },
    Text {
        position: OverlayPoint,
        text: String,
        size: f32,
        color: OverlayColor,
        align: Align2,
        font_family: FontFamily,
    },
}

/// A group of vector primitives sharing an anchor and clipping policy.
#[derive(Clone, Debug)]
pub struct VectorOverlay {
    pub anchor: OverlayAnchor,
    /// Logical-pixel offset from `anchor`.
    pub offset: OverlayPoint,
    pub primitives: Vec<OverlayPrimitive>,
    /// Restrict drawing to the 3D viewport rather than allowing it over application UI.
    pub clip_to_viewport: bool,
}

impl VectorOverlay {
    pub fn new(anchor: OverlayAnchor) -> Self {
        Self {
            anchor,
            offset: (0., 0.),
            primitives: Vec::new(),
            clip_to_viewport: true,
        }
    }
}

fn color((r, g, b, a): OverlayColor) -> Color32 {
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

fn stroke(value: OverlayStroke) -> Stroke {
    Stroke::new(value.width, color(value.color))
}

fn point(origin: Pos2, (x, y): OverlayPoint) -> Pos2 {
    origin + Vec2::new(x, y)
}

fn viewport_rect_points(
    ctx: &Context,
    ui_settings: &UiSettings,
    width: u32,
    height: u32,
    gui_size: (f32, f32),
) -> Rect {
    let pixels_per_pt = ctx.pixels_per_point();
    let logical_width = (width as f32 / pixels_per_pt).round() as u32;
    let logical_height = (height as f32 / pixels_per_pt).round() as u32;
    let (x, y, viewport_width, viewport_height) = viewport_rect(
        gui_size,
        logical_width,
        logical_height,
        ui_settings,
        pixels_per_pt,
    );

    Rect::from_min_size(Pos2::new(x, y), Vec2::new(viewport_width, viewport_height))
}

fn resolve_anchor(
    graphics_state: &GraphicsState,
    overlay: &VectorOverlay,
    viewport: Rect,
    ui_settings: &UiSettings,
    gui_size: (f32, f32),
    pixels_per_pt: f32,
    width: u32,
    height: u32,
) -> Option<Pos2> {
    let origin = match overlay.anchor {
        OverlayAnchor::ViewportTopLeft => viewport.left_top(),
        OverlayAnchor::ViewportTopRight => viewport.right_top(),
        OverlayAnchor::ViewportBottomLeft => viewport.left_bottom(),
        OverlayAnchor::ViewportBottomRight => viewport.right_bottom(),
        OverlayAnchor::ViewportCenter => viewport.center(),
        OverlayAnchor::ViewportPoint((x, y)) => viewport.left_top() + Vec2::new(x, y),
        OverlayAnchor::World(world) => graphics_state.world_to_screen(
            world,
            width,
            height,
            ui_settings,
            gui_size,
            pixels_per_pt,
        )?,
    };

    Some(origin + Vec2::new(overlay.offset.0, overlay.offset.1))
}

pub(crate) fn draw_vector_overlays(
    graphics_state: &GraphicsState,
    ctx: &Context,
    ui_settings: &UiSettings,
    gui_size: (f32, f32),
    width: u32,
    height: u32,
) {
    if graphics_state.scene.vector_overlays.is_empty() {
        return;
    }

    let viewport = viewport_rect_points(ctx, ui_settings, width, height, gui_size);
    let pixels_per_pt = ctx.pixels_per_point();
    let painter = ctx.layer_painter(LayerId::new(
        Order::Background,
        egui::Id::new("vector_overlays"),
    ));

    for overlay in &graphics_state.scene.vector_overlays {
        let Some(origin) = resolve_anchor(
            graphics_state,
            overlay,
            viewport,
            ui_settings,
            gui_size,
            pixels_per_pt,
            width,
            height,
        ) else {
            continue;
        };
        let painter = if overlay.clip_to_viewport {
            painter.with_clip_rect(viewport)
        } else {
            painter.clone()
        };

        for primitive in &overlay.primitives {
            match primitive {
                OverlayPrimitive::Line {
                    start,
                    end,
                    stroke: line_stroke,
                } => {
                    painter.line_segment(
                        [point(origin, *start), point(origin, *end)],
                        stroke(*line_stroke),
                    );
                }
                OverlayPrimitive::Polyline {
                    points,
                    stroke: line_stroke,
                    closed,
                } => {
                    if points.len() < 2 {
                        continue;
                    }
                    let mut points: Vec<_> = points.iter().map(|p| point(origin, *p)).collect();
                    if *closed && points.len() > 1 {
                        points.push(points[0]);
                    }
                    painter.add(Shape::line(points, stroke(*line_stroke)));
                }
                OverlayPrimitive::Circle {
                    center,
                    radius,
                    fill,
                    stroke: circle_stroke,
                } => {
                    let center = point(origin, *center);
                    if let Some(fill) = fill {
                        painter.circle_filled(center, *radius, color(*fill));
                    }
                    if let Some(circle_stroke) = circle_stroke {
                        painter.circle_stroke(center, *radius, stroke(*circle_stroke));
                    }
                }
                OverlayPrimitive::Polygon {
                    points,
                    fill,
                    stroke: polygon_stroke,
                } => {
                    if points.len() < 3 {
                        continue;
                    }
                    let points = points.iter().map(|p| point(origin, *p)).collect();
                    painter.add(Shape::convex_polygon(
                        points,
                        color(*fill),
                        polygon_stroke.map(stroke).unwrap_or(Stroke::NONE),
                    ));
                }
                OverlayPrimitive::Text {
                    position,
                    text,
                    size,
                    color: text_color,
                    align,
                    font_family,
                } => {
                    painter.text(
                        point(origin, *position),
                        *align,
                        text,
                        FontId::new(*size, font_family.clone()),
                        color(*text_color),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_point_is_relative_to_overlay_origin() {
        assert_eq!(point(Pos2::new(12., 34.), (-2., 6.)), Pos2::new(10., 40.));
    }
}
