# Demand Generation Tools

## extract_cluster_edges.py

Extract edge IDs per OD cluster polygon from a SUMO network (`.net.xml`).

This tool is intended to support synthetic ride-pooling demand generation by building cluster-specific edge pools that can later be sampled as origins/destinations.

### What it does

1. Loads cluster polygons from a text file (for example `clusters.txt`).
2. Loads SUMO network via `sumolib`.
3. Extracts edge geometry with robust fallback:
   1. `edge.getShape()`
   2. first lane shape (`edge.getLanes()[0].getShape()`)
   3. from/to node coordinates (`edge.getFromNode().getCoord()`, `edge.getToNode().getCoord()`)
4. Applies filtering (internal edges, length, permissions).
5. Assigns edges to clusters using selected membership mode.
6. Writes outputs (JSON, txt files, summary CSV, optional visualization).

### Cluster file format

Example (`demand_generation/clusters.txt`):

```txt
# format:
# [Cluster name]: (left bottom),(right bottom),(right top),(left top)
# coordinates are pos attribute of net.xml
A: (0,900),(900,900),(900,1600),(0,1600)
B: (1900,1800),(2900,1800),(2900,2600),(1900,2600)
C: (1900,0),(2600,0),(2600,900),(1900,900)
```

Rules:
- Empty lines are ignored.
- Lines starting with `#` are ignored.
- Cluster name is parsed before `:`.
- Coordinate pairs are parsed after `:`.
- At least 3 coordinate pairs are required.
- Coordinates are SUMO Cartesian coordinates.

### Membership modes

Use `--membership`:
- `midpoint`: edge midpoint is inside polygon
- `any-point` (default): any geometry point is inside polygon
- `segment-intersects`: any polyline segment intersects polygon boundary or interior

### Filtering options

- `--exclude-internal` / `--include-internal` (default: exclude)
- `--min-length FLOAT` (default: `0.0`)
- `--max-length FLOAT`
- `--allow-vclass VCLASS` (repeatable)
- `--disallow-vclass VCLASS` (repeatable)
- `--require-drivable` / `--no-require-drivable` (default: require)
- `--deduplicate` / `--no-deduplicate` (default: deduplicate)

### Outputs

- `--out-json FILE`: cluster to edge-ID mapping JSON
- `--out-dir DIR`: one text file per cluster (`<cluster>.txt`)
- `--summary-csv FILE`: per-cluster summary CSV
- `--viz-add PATH`: visualization output
  - `--viz-format add`: write SUMO additional file with colored polylines
  - `--viz-format selection`: write per-cluster `.sel.xml` files

If no output options are provided, JSON is printed to stdout.

### Visualization options

For `--viz-format add`:
- `--cluster-colors A=#ff0000,B=#00ff00,...`
- `--viz-line-width FLOAT` (default: `6.0`)
- `--viz-layer INT` (default: `120`)

Default palette is bright/high-contrast to improve visibility.

### Overlap policy

If an edge belongs to multiple clusters, use `--overlap-policy`:
- `first`: assign only to first matched cluster
- `allow`: include in all matched clusters
- `warn` (default): include in all and print warning
- `error`: fail on overlap

### Strict mode and logging

- `--strict` enforces hard failures for:
  - malformed cluster lines
  - empty clusters
  - ambiguous permission checks
  - overlap when `--overlap-policy error`
- `--verbose` enables extra diagnostics.

The script always prints a summary:
- clusters loaded
- total edges
- edges after filtering
- selected edges per cluster
- geometry fallback usage

### Prerequisites

`sumolib` must be importable.

Typical setup:
- Install SUMO.
- Ensure `SUMO_HOME` is set.
- Add `$SUMO_HOME/tools` to `PYTHONPATH`.

### Example commands

Full extraction and visualization:

```bash
python demand_generation/extract_cluster_edges.py \
  --net configs/randgrid.net.xml \
  --clusters demand_generation/clusters.txt \
  --out-json demand_generation/clusters_edges.json \
  --out-dir demand_generation/clusters_edges \
  --summary-csv demand_generation/clusters_summary.csv \
  --viz-add demand_generation/clusters_viz.add.xml \
  --viz-format add \
  --membership any-point \
  --exclude-internal \
  --min-length 10 \
  --allow-vclass passenger \
  --allow-vclass taxi
```

Higher-visibility overlays:

```bash
python demand_generation/extract_cluster_edges.py \
  --net configs/randgrid.net.xml \
  --clusters demand_generation/clusters.txt \
  --viz-add demand_generation/clusters_viz.add.xml \
  --viz-format add \
  --allow-vclass passenger --allow-vclass taxi \
  --viz-line-width 9 \
  --viz-layer 140
```

Selection-file visualization:

```bash
python demand_generation/extract_cluster_edges.py \
  --net configs/randgrid.net.xml \
  --clusters demand_generation/clusters.txt \
  --viz-add demand_generation/clusters_viz \
  --viz-format selection
```

### Output files generated in this folder

- `demand_generation/clusters_edges.json`
- `demand_generation/clusters_edges/*.txt`
- `demand_generation/clusters_summary.csv`
- `demand_generation/clusters_viz.add.xml`

These edge lists can be reused by downstream OD or passenger-wave generators.

## generate_wave_demand.py

Generate wave-based SUMO `<person>` taxi demand from cluster edge lists.

This script consumes a cluster-to-edge mapping JSON (for example produced by `extract_cluster_edges.py`) and generates passenger requests in waves using built-in OD templates.

### Main behavior

1. Load cluster edge pools from `--clusters-json`.
2. Build waves over simulation time:
   1. sample wave interval and size,
   2. pick current template from template cycle,
   3. split passengers into structured and random trips,
   4. sample `from`/`to` edges for each passenger (`fromEdge != toEdge`).
3. Write SUMO routes XML with `<person><ride .../></person>`.
4. Optionally write a CSV with one row per passenger.

### Default setup

- `--start-time 0`
- `--end-time 2400`
- `--num-waves 20`
- `--wave-size-min 6`
- `--wave-size-max 8`
- `--wave-interval-min 90`
- `--wave-interval-max 150`
- `--random-trip-share 0.10`
- `--template-cycle AB_AC,BA_CA,AB_BC,CB_BA,AC_BC,CA_CB`
- `--random-trip-mode cluster-edges`
- `--seed 42`

Wave generation stops when:
- requested number of waves is reached, or
- next wave departure would be `>= end_time`.

### Built-in templates

Pure templates:
- `AB`, `BA`, `AC`, `CA`, `BC`, `CB`

Mixed templates:
- `AB_AC` (50% A->B, 50% A->C)
- `BA_CA` (50% B->A, 50% C->A)
- `AB_BC` (50% A->B, 50% B->C)
- `CB_BA` (50% C->B, 50% B->A)
- `AC_BC` (50% A->C, 50% B->C)
- `CA_CB` (50% C->A, 50% C->B)

### Random trip modes

- `cluster-edges` (default): sample random edges globally from all cluster edges.
- `cluster-pairs`: sample random origin cluster and destination cluster, then sample edges inside each cluster.

### Parking area behavior

By default, each ride includes `parkingArea` derived from destination edge:

- `parkingArea = parking_area_prefix + to_edge`
- default prefix: `pa`

Examples:
- `to="-3096"` -> `parkingArea="pa-3096"`
- `to="-37"` -> `parkingArea="pa-37"`
- `to="2502"` -> `parkingArea="pa2502"`

Controls:
- `--include-parking-area` / `--no-include-parking-area`
- `--parking-area-prefix`

### Validation and summary

The script validates:
- template clusters exist and are non-empty,
- `wave_size_min <= wave_size_max`,
- `start_time < end_time`,
- `0 <= random_trip_share <= 1`,
- interval bounds are valid.

It warns if requested `num_waves` is likely too high for the time window and minimum interval.

It prints summary stats:
- number of waves and passengers,
- random vs structured passenger counts,
- counts per template,
- counts per OD relation,
- first and last departure times.

### Outputs

Required:
- `--out-xml`

Optional:
- `--out-csv`

CSV columns:
- `person_id,wave_idx,depart,trip_type,template,origin_cluster,destination_cluster,from_edge,to_edge,parking_area`

### Example usage

Generate with defaults:

```bash
python demand_generation/generate_wave_demand.py \
  --clusters-json demand_generation/clusters_edges.json \
  --out-xml demand_generation/wave_demand.xml
```

Generate with CSV and explicit template cycle:

```bash
python demand_generation/generate_wave_demand.py \
  --clusters-json demand_generation/clusters_edges.json \
  --out-xml demand_generation/wave_demand.xml \
  --out-csv demand_generation/wave_demand.csv \
  --template-cycle AB_AC,AB_BC,AC_BC,AB,AC,BC \
  --random-trip-share 0.10 \
  --seed 42
```

Generate with parameters of waves specified:

```bash
python demand_generation/generate_wave_demand.py \
  --clusters-json demand_generation/clusters_edges.json \
  --out-xml demand_generation/wave_demand.xml \
  --num-waves 10 \
  --wave-size-min 4 --wave-size-max 5 \
  --wave-interval-min 160 --wave-interval-max 230 \
  --random-trip-share 0.05
```

## generate_taxi_route_file.py

Create a complete SUMO route file (`.rou.xml`) containing:
- taxi `vType` with taxi device params,
- generated taxi `<trip>` entries,
- imported passenger `<person>` entries.

This script is intended as the final composition step for ride-pooling runs after passenger demand generation.

### Main behavior

1. Optionally load cluster edge pools from `--clusters-json`.
2. Generate taxi start/destination edges according to init/destination modes.
3. Optionally import passengers from `--passengers-xml`.
4. Validate and optionally repair passenger `parkingArea` attributes.
5. Write a single combined `.rou.xml` file.
6. Optionally export a taxi manifest CSV.

### Taxi type and defaults

Taxi defaults are:
- `--taxi-capacity 2`
- `--taxi-color green`
- `--pickup-duration 0`
- `--dropoff-duration 10`
- `--taxi-parking` enabled
- `--taxi-depart 0`
- `--taxi-id-prefix t`

Generated `vType`:
- `id="taxi"`
- `vClass="taxi"`
- `personCapacity=<taxi-capacity>`
- `<param key="has.taxi.device" value="true"/>`
- `<param key="device.taxi.pickUpDuration" .../>`
- `<param key="device.taxi.dropOffDuration" .../>`
- `<param key="device.taxi.parking" value="true|false"/>`

### Taxi initialization modes

Use `--taxi-init-mode`:
- `cluster-balanced` (default): distribute taxis as evenly as possible over `--taxi-init-clusters`.
- `cluster-random`: randomly assign each taxi to one cluster from `--taxi-init-clusters`.
- `edge-list`: read initial edges from `--taxi-from-edges-file` (one edge per line).
- `manual`: read initial edges from `--taxi-from-edges` (comma-separated).

Notes:
- Cluster-based modes require `--clusters-json`.
- If provided edge count is smaller than `--num-taxis`, values are cycled unless `--strict` is set.

### Taxi destination modes

Use `--taxi-dest-mode`:
- `random-valid` (default): sample destination from all cluster edges with `to != from` when possible.
- `cluster-random`: sample destination from a cluster pool (prefer taxi init cluster when available).
- `same-as-from`: set `to == from` intentionally (warning emitted).
- `manual`: set destinations from `--taxi-to-edges` or `--taxi-to-edges-file`.

Notes:
- `random-valid` and `cluster-random` require `--clusters-json`.
- `manual` requires one of `--taxi-to-edges` or `--taxi-to-edges-file`.

### Passenger import and parkingArea handling

Passenger source:
- `--passengers-xml FILE` imports `<person>` entries.
- If XML root is `<routes>`, direct child `<person>` elements are imported.

Parking behavior:
- `--include-parking-area` (default): keep/expect passenger `parkingArea`.
- `--no-include-parking-area`: do not enforce `parkingArea` presence.
- `--repair-parking-areas`: if enabled together with include mode, missing `parkingArea` is repaired as:
  - `parkingArea = <parking-area-prefix> + <ride to edge>`
- `--parking-area-prefix` default is `pa`.

Strict mode:
- `--strict` converts multiple warnings into hard errors, including malformed cluster selections, missing required pools/edges, and missing required parking data.

### Outputs

Required:
- `--out-rou FILE`
- `--num-taxis INT`

Optional:
- `--out-taxi-csv FILE` taxi manifest with columns:
  - `taxi_id,depart,from_edge,to_edge,init_cluster,capacity`

The script prints a summary including:
- output file path,
- taxi count and capacity,
- initial cluster counts,
- passenger count,
- `parkingArea` coverage (`rides_with_parking / rides_total`),
- warnings.

### Example commands

Cluster-balanced taxis + imported passenger waves:

```bash
python demand_generation/generate_taxi_route_file.py \
  --clusters-json demand_generation/clusters_edges.json \
  --passengers-xml demand_generation/wave_demand.xml \
  --out-rou configs/wave_demand_cap2_taxis6.rou.xml \
  --num-taxis 6 \
  --taxi-capacity 2 \
  --taxi-init-mode cluster-balanced \
  --taxi-init-clusters A,B,C \
  --taxi-dest-mode random-valid \
  --seed 42 \
  --out-taxi-csv demand_generation/wave_demand_cap2_taxis6_taxis.csv
```

Manual from/to edge lists:

```bash
python demand_generation/generate_taxi_route_file.py \
  --out-rou configs/manual_taxis.rou.xml \
  --num-taxis 4 \
  --taxi-init-mode manual \
  --taxi-from-edges "-1001,-1002,-1003,-1004" \
  --taxi-dest-mode manual \
  --taxi-to-edges "-2001,-2002,-2003,-2004"
```

Edge-list initialization with strict validation:

```bash
python demand_generation/generate_taxi_route_file.py \
  --clusters-json demand_generation/clusters_edges.json \
  --passengers-xml demand_generation/wave_demand_test.xml \
  --out-rou configs/strict_taxis.rou.xml \
  --num-taxis 8 \
  --taxi-init-mode edge-list \
  --taxi-from-edges-file demand_generation/taxi_from_edges.txt \
  --taxi-dest-mode cluster-random \
  --taxi-init-clusters A,B,C \
  --repair-parking-areas \
  --strict
```
