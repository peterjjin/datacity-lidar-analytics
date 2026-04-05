# Autoware customizations

This directory stores only local Autoware changes that should be versioned in this repository.

It does not vendor the full upstream Autoware source tree.

Current assumptions:

- External workspace root: `/home/datacity/autoware_core_workspace`
- Upstream source repo: `/home/datacity/autoware_core_workspace/src/autoware_core`
- Upstream remote: `https://github.com/autowarefoundation/autoware_core.git`

## Current tracked customization

- `patches/autoware_core_lanelet2_map_visualizer.patch`

This patch changes the lanelet2 map visualizer to render obstacle-removal polygons using:

- `lanelet::visualization::obstaclePolygonsAsMarkerArray(...)`

instead of:

- `lanelet::visualization::obstacleRemovalAreaAsMarkerArray(...)`

## Apply the tracked customization

```bash
cd /home/datacity/lidar-analytics
bash autoware_custom/apply_autoware_customizations.sh
```

## Refresh the tracked patch from the live workspace

If you make additional edits in `/home/datacity/autoware_core_workspace/src/autoware_core`, export the updated patch with:

```bash
cd /home/datacity/lidar-analytics
bash autoware_custom/export_autoware_customizations.sh
```

Then commit the updated patch file in this repo.
