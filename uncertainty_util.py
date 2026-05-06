import numpy as np
from scipy.ndimage import binary_erosion, center_of_mass

def extract_bands(seg, unc_map):
    """
    Function that extracts:
    - seg_edge: the edge of the segmentation
    - unc_inner: the inner part of the uncertainty band (uncertainty pixels that are inside the segmentation, excluding the edge)
    - unc_outer: the outer part of the uncertainty band (uncertainty pixels that are outside the segmentation)
    - unc_edge_inner: the inner edge of the uncertainty band (uncertainty pixels on the inner edge of the segmentation)
    - unc_edge_outer: the outer edge of the uncertainty band (uncertainty pixels on the outer edge of the segmentation)
    """

    #Change maps into boolean for easier operations
    seg = seg.astype(bool)
    unc_map = unc_map.astype(bool)

    seg_edge = seg & ~binary_erosion(seg) #Finds edge of the segmentation by isolating eroded pixels

    unc_inner = unc_map & seg & ~seg_edge #Finds all uncertainty pixels on the inside of the segmentation, excluding the edge
    unc_outer = unc_map & ~seg #Finds all uncertainty pixels on the outside of the segmentation

    unc_band_edges = binary_erosion(unc_map)
    
    unc_edge_inner = (unc_inner & ~binary_erosion(unc_inner)) & ~unc_band_edges #Inner edge of the uncertainty band
    unc_edge_outer = (unc_outer & ~binary_erosion(unc_outer)) & ~unc_band_edges #Outer edge of the uncertainty

    return seg_edge, unc_inner, unc_outer, unc_edge_inner, unc_edge_outer

def order_segmentation_pixels(seg_edge):
    """
    TODO: MAKE FUNCTION DESCRIPTION
    """

    seg_edge = seg_edge.astype(bool)

    coords = np.argwhere(seg_edge)

    if coords.shape[0] == 0:
        return []

    # Store pixels as (y, x)
    unvisited = set(map(tuple, coords))

    start = tuple(coords[0])
    current = start

    ordered_edge_pixels = [current]
    unvisited.remove(current)

    kernel = [(0, 1),(1, 0),(0, -1),(-1, 0),(1, 1),(1, -1),(-1, -1),(-1, 1),]

    while len(unvisited) > 0:
        found_next = False

        y, x = current

        for dy, dx in kernel:
            next_pixel = (y + dy, x + dx)

            if next_pixel in unvisited:
                ordered_edge_pixels.append(next_pixel)
                unvisited.remove(next_pixel)
                current = next_pixel
                found_next = True
                break

        if not found_next:
            break

    return np.array(ordered_edge_pixels)

def determine_band_thickness_mm_raycast(
    seg,
    unc_map,
    unc_inner,
    seg_edge,
    unc_outer,
    pixel_spacing=(1.0, 1.0, 1.0),
    angle_step=1,
    step_mm=None,
    pad=5,
):
    """
    TODO: MAKE FUNCTION DESCRIPTION
    """

    #Set up masks for raycasting, convert to boolean for easier operations
    seg = seg.astype(bool)
    unc_map = unc_map.astype(bool)
    unc_inner = unc_inner.astype(bool)
    seg_edge = seg_edge.astype(bool)
    unc_outer = unc_outer.astype(bool)

    #Extract pixel spacing
    spacing_y = pixel_spacing[1]
    spacing_x = pixel_spacing[2]

    #Basic sanity checks-
    if not seg.any():
        raise ValueError("Segmentation is empty.")
    if not unc_map.any():
        raise ValueError("Uncertainty map is empty.")

    #Set step size for raycasting, is 1/4th of pixel spacing consistent with local normals method
    if step_mm is None:
        step_mm = min(spacing_y, spacing_x) * 0.25

    #center of mass of segmentation
    cy, cx = center_of_mass(seg)
    h, w = seg.shape

    #Compute bounding box round uncertainty region to make sure rays are not sampled throughout entire image
    ys, xs = np.where(unc_map)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    #center of bbox
    cy_box = (y_min + y_max) / 2
    cx_box = (x_min + x_max) / 2

    #square bbox around uncertainty region (with padding)
    half_size = max(y_max - y_min, x_max - x_min) / 2
    half_size = int(np.ceil(half_size)) + pad

    y0 = max(0, int(np.floor(cy_box - half_size)))
    y1 = min(h, int(np.ceil(cy_box + half_size + 1)))
    x0 = max(0, int(np.floor(cx_box - half_size)))
    x1 = min(w, int(np.ceil(cx_box + half_size + 1)))

    #Convert bbox to mm coords for ray clipping
    y0_mm = y0 * spacing_y
    y1_mm = (y1 - 1) * spacing_y
    x0_mm = x0 * spacing_x
    x1_mm = (x1 - 1) * spacing_x

    #convert center of mass to mm coords
    cy_mm = cy * spacing_y
    cx_mm = cx * spacing_x

    #Set maximum ray length for computational efficiency
    max_length_mm = np.sqrt((h * spacing_y) ** 2 + (w * spacing_x) ** 2)

    #set all angles of the rays to be cast
    angles = np.arange(0, 360, angle_step)

    #create storage lists for results
    seg_mm = []
    inner_mm = []
    edge_mm = []
    outer_mm = []
    band_total_mm = []

    #Loop over every angle to be sampled
    for angle_deg in angles:
        theta = np.deg2rad(angle_deg)

        #convert direction of ray to mm space
        dy_mm = np.sin(theta)
        dx_mm = np.cos(theta)

        #Distances along the ray
        distances = np.arange(0, max_length_mm, step_mm)

        #change ray coordintes to mm
        ray_y_mm = cy_mm + distances * dy_mm
        ray_x_mm = cx_mm + distances * dx_mm

        #Clip the ray to remain inside decided bbox
        valid = (
            (ray_y_mm >= y0_mm) & (ray_y_mm <= y1_mm) &
            (ray_x_mm >= x0_mm) & (ray_x_mm <= x1_mm)
        )

        ray_dist_mm = distances[valid]
        ray_y_mm = ray_y_mm[valid]
        ray_x_mm = ray_x_mm[valid]

        #conver to pixel indices
        ray_y = ray_y_mm / spacing_y
        ray_x = ray_x_mm / spacing_x

        iy = np.round(ray_y).astype(int)
        ix = np.round(ray_x).astype(int)

        #Keep only valid indices inside the image
        inside = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
        iy = iy[inside]
        ix = ix[inside]
        ray_dist_mm = ray_dist_mm[inside]

        #sample along the rays
        seg_hits = seg[iy, ix]
        inner_hits = unc_inner[iy, ix]
        edge_hits = seg_edge[iy, ix]
        outer_hits = unc_outer[iy, ix]

        #convert from pixel space to mm space
        seg_len = np.sum(seg_hits) * step_mm
        inner_len = np.sum(inner_hits) * step_mm
        edge_len = np.sum(edge_hits) * step_mm
        outer_len = np.sum(outer_hits) * step_mm

        #compute total band length to store as seperate value (nice? i guess? haha?)
        total_len = inner_len + edge_len + outer_len

        #store final results
        seg_mm.append(seg_len)
        inner_mm.append(inner_len)
        edge_mm.append(edge_len)
        outer_mm.append(outer_len)
        band_total_mm.append(total_len)


    return {
        "center_of_mass_px": (cy, cx),
        "center_of_mass_mm": (cy_mm, cx_mm),
        "angles_deg": angles,
        "seg_mm": np.array(seg_mm),
        "inner_mm": np.array(inner_mm),
        "edge_mm": np.array(edge_mm),
        "outer_mm": np.array(outer_mm),
        "total_mm": np.array(inner_mm) + np.array(outer_mm),
        "band_total_mm": np.array(band_total_mm),
        "bbox_square_px": (y0, y1, x0, x1),
        "step_mm": step_mm,
    }

def determine_band_thickness_mm_normals(
    seg,
    unc_inner,
    seg_edge,
    unc_outer,
    ordered_edge_pixels,
    interpix_dist=2,
    pixel_interval=1,
    pixel_spacing=(1.0, 1.0,1.0)
):
    """
    TODO: MAKE FUNCTION DESCRIPTION
    """

    ordered_edge_pixels = np.asarray(ordered_edge_pixels, dtype=float)
    sy, sx = float(pixel_spacing[1]), float(pixel_spacing[2])
    spacing_yx = np.array([sy, sx], dtype=float)
    step_mm = min(sy, sx) * 0.25
    n_pixels = len(ordered_edge_pixels)
    H, W = seg.shape

    if n_pixels == 0:
        return []

    pixel_interval = max(1, int(pixel_interval))
    interpix_dist = max(1, int(interpix_dist))

    res_pixel_index = []
    res_pixel_yx = []
    res_tangent_yx = []
    res_outer_normal_yx = []
    res_inner_normal_yx = []
    res_outer_mm = []
    res_inner_mm = []
    res_total_mm = []

    results = []

    for pixel_index in range(0, n_pixels, pixel_interval):
        
        y, x = ordered_edge_pixels[pixel_index]
        pixel_yx = np.array([y, x], dtype=float)

        #Compute the begin and end point to compute tangent vector, % for wrap-around indexing
        idx1 = (pixel_index - interpix_dist) % n_pixels
        idx2 = (pixel_index + interpix_dist) % n_pixels

        tangent = ordered_edge_pixels[idx2] - ordered_edge_pixels[idx1]

        norm = np.linalg.norm(tangent)
        if norm == 0:
            continue

        tangent = tangent / norm

        #Because pixels are ordered, outer and inner normals are defined as
        outer_normal = np.array([-tangent[1], tangent[0]])
        inner_normal = np.array([tangent[1], -tangent[0]])

        def measure_band_mm(seg_edge, start_yx, normal_yx, band, max_search_mm=20.0):
            """
            March from start_yx along normal_yx and measure how many mm remain inside band.
            normal_yx is in pixel coordinates.
            """

            normal_yx = np.asarray(normal_yx, dtype=float)
            normal_mm = normal_yx * spacing_yx

            normal_mm_norm = np.linalg.norm(normal_mm)
            if normal_mm_norm == 0:
                return 0.0

            # Convert a physical step in mm back to pixel-space step
            step_yx = (normal_mm / normal_mm_norm) * (step_mm / spacing_yx)

            dist_mm = 0.0
            pos_yx = np.asarray(start_yx, dtype=float).copy()

            while dist_mm < max_search_mm:
                pos_yx = pos_yx + step_yx
                dist_mm += step_mm

                yy = int(round(pos_yx[0]))
                xx = int(round(pos_yx[1]))

                if yy < 0 or yy >= H or xx < 0 or xx >= W:
                    break

                if not (band[yy, xx] or seg_edge[yy,xx]):
                    break

            return dist_mm
        
        outer_mm = measure_band_mm(
            seg_edge=seg_edge,
            start_yx=pixel_yx,
            normal_yx=outer_normal,
            band=unc_outer,
        )

        inner_mm = measure_band_mm(
            seg_edge=seg_edge,
            start_yx=pixel_yx,
            normal_yx=inner_normal,
            band=unc_inner,
        )

        total_mm = inner_mm + outer_mm

        res_pixel_index.append(pixel_index)
        res_pixel_yx.append(pixel_yx)
        res_tangent_yx.append(tangent)
        res_outer_normal_yx.append(outer_normal)
        res_inner_normal_yx.append(inner_normal)
        res_outer_mm.append(outer_mm)
        res_inner_mm.append(inner_mm)
        res_total_mm.append(total_mm)

    return {
            "pixel_index": res_pixel_index,
            "pixel_yx": res_pixel_yx,
            "tangent_yx": res_tangent_yx,
            "outer_normal_yx": res_outer_normal_yx,
            "inner_normal_yx": res_inner_normal_yx,
            "outer_mm": res_outer_mm,
            "inner_mm": res_inner_mm,
            "total_mm": res_total_mm,
        }





