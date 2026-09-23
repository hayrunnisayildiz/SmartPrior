# Keivitsa (GTK) drillhole tables — Stage 0

Inspection only. No sample adapter has been written. `KEIVITSA_ROOT` was unset; the tables below were read from the supplied archive at `database/keivitsa/source/gtk`. `processed/` and `exports/` were not used. Nothing from that archive is in this repository. Measured numbers below are summary statistics. Example rows are schematic.

## Files read

Caret text is Latin-1, field separator `^`, then three metadata rows (storage type, width, decimals) before the data. Shapefile `.prj` files are ESRI WKT. Row counts are live records (no deleted dBase rows in these tables).

| Table | Path under the GTK root | Rows | Holes |
|---|---|---:|---:|
| Collars | `report/3_DRILLINGS/Logs/Shape_files/collar.shp` (+ `.dbf`, `.prj`) | 522 | 522 |
| Survey | `report/3_DRILLINGS/Logs/Shape_files/kalte.txt` | 3,621 | 522 |
| Survey, same stations | `report/3_DRILLINGS/Logs/Shape_files/survey.dbf` | 3,621 | 522 |
| Cu, method 511P | `report/3_DRILLINGS/Assays/Shape_files/511P.txt` and `d511p.dbf` | 16,004 | 289 |
| Method catalogue | `report/3_DRILLINGS/Assays/Shape_files/analy.txt` | 27 methods | — |
| Petrophysics | `report/3_DRILLINGS/Downhole_soundings_and_core_measurements/petro.txt` and `petroph.dbf` | 63,001 | 264 |
| Hole-id lists | `tunnus.txt` beside the collars, assays, and petrophysics | 522 ids each | 522 |

`analy.txt` counts 16,004 samples for method `511P`, matching both assay files. An Excel copy `Assays/Excel_files/d511p.xls` is present and was not parsed. Other assay workbooks exist (28). Cu also appears in `d511a` (method 511A); `d511u` has no Cu column.

Documentation used:

- `report/6_DESCRIPTION/0_DATA_SETS/Description of data sets.doc` — folder contents.
- `report/6_DESCRIPTION/1_DRILLINGS/Liite_MRVK_3-O2_kairaus_GTK.doc` — GTK drilling practice (collars, inclination, depth).
- `report/6_DESCRIPTION/2_ANALYSES/Meth_511P_511U.doc` — method 511P, Cu detection limit.
- `report/6_DESCRIPTION/2_ANALYSES/General.doc` — method-code list; `ppm` definition.
- `collar.prj` (and the assay and petrophysics `.prj` files, same WKT).
- `database/keivitsa/source/geofysiikan_opetusmateriaali.html` — coordinate-system name on the geophysical delivery.

## Schematic rows

Coordinates and assay numbers are omitted. Flags and column roles are the real ones.

Collar (`collar.dbf`): `HOLE_ID`, `DH_TYPE`, `YEAR`, `DEPARTMENT`, `YKJ_NORTH`, `YKJ_EAST`, `KKJ_NORTH`, `KKJ_EAST`, `Z`, `LENGTH`, `STARTAZIM`, `STARTDIP`, `POS_ACCUR`, `DESC`. One row per hole. `DH_TYPE` is `SYVÄKAIRAUS` on every row.

Survey (`kalte.txt`): `Tunnus`, `Yht_X`, `Yht_Y`, `Kkj_X`, `Kkj_y`, `Z`, `Syvyys`, `Kaltevuus`, `Suunta`. `survey.dbf` is the same stations with English names `HOLE_ID`, `KKJ_NORTH`, `KKJ_EAST`, `Z`, `DEPTH`, `DIP`, `AZIMUTH`. All 3,621 stations match on depth, dip, and azimuth.

Assay 511P: `Tunnus` / `HOLE_ID`, `Ylasyvyys` / `FROM`, `Alasyvyys` / `TO`, collar coordinates repeated as `ReiKkj_X` / `KKJ_NORTH` and `ReiKkj_y` / `KKJ_EAST`, `Cu`, `Cu_L` / `CU_L`. `analy.txt` line 1 is `Oletusyksikko: ppm`.

Petrophysics: `Tunnus` / `HOLE_ID`, `Syvyys` / `DEPTH`, then `PTR_D`, `PTR_J`, `PTR_K`, `LUO_R`, `DSR_D`, `DSR_J`, `DSR_K`. A missing cell is `*************`, not a blank.

## Conventions

### a) Coordinate system and axis order

`collar.prj` is `PROJCS["Finland_Zone_3"]`, datum `D_KKJ`, International 1924, Gauss–Krüger, central meridian 27°, false easting 3,500,000 m, false northing 0. That parameter set is KKJ / Finland zone 3, EPSG:2393. The assay and petrophysics `.prj` files are the same string. The geophysics HTML names the delivery “Kartastokoordinaattijärjestelmä KKJ/zone 3”.

The collar table names the axes. On all 522 collars, `KKJ_NORTH` is the 7×10⁶ m band and `KKJ_EAST` is the 3×10⁶ m band (the false-easting band). In `kalte.txt`, `Kkj_X` equals `KKJ_NORTH` and `Kkj_y` equals `KKJ_EAST` at the first station of all 522 holes. YKJ (`YKJ_NORTH` / `YKJ_EAST`, caret names `Yht_X` / `Yht_Y`) differs from KKJ by at most 1 m, which is the integer-metre storage of those columns.

Shapefile geometry is the other way around: geometry X matches `KKJ_EAST` and geometry Y matches `KKJ_NORTH` on all 522 collars (point type 1, no Z). Ground gravity, ground magnetics, and the airborne magnetic XYZ also store the 3×10⁶ m value in column 1 and the 7×10⁶ m value in column 2 (`X/m Y/m` in the ground-survey headers; `X Y` in the airborne header). Same CRS, easting-then-northing column order. The drillhole attribute `Kkj_X` is northing.

### b) Angle units

Degrees, not gon.

The drilling note, section 4.2, says inclination is measured to 0.1 degree (`0.1 asteen tarkkuudella`). In `kalte.txt`, `Suunta` spans 0 to 360 inclusive, with no value above 360. A gon azimuth would be allowed to reach 400. `Kaltevuus` spans 38.1 to 90. Collar `STARTAZIM` has the same 0–360 span.

### c) Azimuth reference and dip sign

Dip sign is fixed by the measurements. `Kaltevuus` / `DIP` has no negative value, no zero, and no value above 90. 685 of 3,621 stations are exactly 90, and 90 is the maximum. Collar `STARTDIP` is exactly 90 on 340 of 522 holes. Two collars have `STARTDIP` 0; both have survey inclination 90, so a collar 0 is an empty plan value. Vertical is +90 and the hole is downward. This is inclination from horizontal, not the mining convention in which a downward hole is −90.

Azimuth reference is not stated in the drilling note. The note says a local grid is tied to the national zone grid, and that the departure direction is `suunta` and `kaltevuus` (sections 2.4 and 4.2). It does not say grid north versus magnetic north, or clockwise versus counter-clockwise. In the survey, 3,529 of 3,621 azimuths are an exact multiple of 90° (0, 90, 180, 270, or 360). That is what grid-planned holes look like. It does not by itself prove the sense of rotation. See the questions at the end.

### d) Survey depth

`Syvyys` / `DEPTH` is along-hole length from the collar.

521 of 522 holes have a station at 0. The other hole’s first station is at 10.0 m and its deepest station equals `LENGTH`. The most common step is 10.0 m (2,545 of the steps), which is the 10 m inclination interval in drilling-note section 4.2. On 504 of 522 holes, `LENGTH` and the deepest `Syvyys` agree within 0.05 m; on 510 they agree within 1 m. Twelve holes differ by more than 1 m (largest gap 87.7 m). Collar coordinates and collar `Z` are constant on every station of every hole, so the survey file is not a desurveyed trajectory.

Station counts run from 2 to 63. 65 holes change inclination by more than 2°.

### e) Collar elevation

Field `Z` (caret `Z`, two decimal places). The drilling note, section 4.2, says the collar is checked at the casing–ground intersection by levelling or RTK GPS, with a stated requirement of 0.1 m in Z and 1 m in X and Y. Non-zero elevations span 203 to 312.7 m (median 231.5 m, 389 holes).

The height-system name (N60 or N2000) is not in the note or the table. Collar years are 1984–1995.

133 collars have `Z = 0`. That set is exactly `DEPARTMENT = K` and `POS_ACCUR = 10`. The other collars are department `M`, with `POS_ACCUR` 1 (1 hole), 2 (288), 3 (74), or 99 (26). `POS_ACCUR` is not defined in the drilling note. No collar has a KKJ coordinate outside the zone-3 bands above, so the failure on this set is the elevation, not the plan position.

Eight of the `Z = 0` collars have 511P samples: `K371494R123`, `K371494R140`, `K371494R145`, `K371494R167`, `K371495R170`, `K371495R204`, `K371495R216`, `K371495R219`. No elevation will be invented for them.

### f) Cu method 511P and below-detection encoding

`Meth_511P_511U.doc`: 511P is ICP-AES on an aqua-regia digest; the listed Cu detection limit is 1 ppm. 511U is GFAAS and the Cu line is not in that element list. `General.doc` gives the same method names and states `ppm = µg/g`. `analy.txt` repeats that the default unit is ppm.

511P Cu counts:

- 16,004 rows, 289 holes, every row has a Cu cell.
- 15,915 rows have an empty `CU_L` and a positive Cu. The smallest unflagged value is 0.5 ppm, and 41 unflagged values are ≤ 1 ppm. Those 41 are stored as measurements.
- `CU_L = "<"` on 72 rows. The stored number is 0.5000 (40 rows) or 1.0000 (32 rows).
- `CU_L = "!"` on 17 rows: 9 zeros and 8 positive values ≤ 1 ppm.
- No negative Cu, and the value cell itself never contains a `<` token.

Across all element flag columns in `d511p`, the observed flag characters are `!`, `-`, `<`, and `>`. Cu in this file uses only `<` and `!`. Method 511A (`d511a.dbf`) has another 2,361 Cu rows on 85 holes, of which 74 holes are absent from 511P. Its Cu flags are `-` (38 rows, no numeric Cu) and `>` (8 rows). Those rows stay out of the 511P count.

There are no duplicate `FROM`–`TO` intervals and no overlapping intervals in 511P. Interval length is 0.1–10 m (median 2 m).

### g) Density and magnetic susceptibility

Both live in `petro.txt` / `petroph.dbf`, folder “Downhole soundings and core measurements”. Missing cells are `*************`.

| Columns | Numeric rows | Holes | Spacing of numeric depths | Numeric range |
|---|---:|---:|---|---|
| `PTR_D`, `PTR_J`, `PTR_K` | 10,412 (the three columns always together) | 122 | median 1 m (min 0.05 m) | `PTR_D` 2,137–4,521 |
| `DSR_D`, `DSR_J`, `DSR_K` | 20,496 (always together) | 141 | median 1 m (min 1 m) | `DSR_D` 2,271–4,186 |
| `LUO_R` | 32,382 | 44 | median 0.33 m | 0.11–1.24×10⁶ |

The PTR holes and the DSR holes are disjoint. `PTR_D` / `DSR_D` have no non-positive values. The numbers are thousands, so the unit is kg/m³ (a density in g/cm³ cannot be 3,139). `PTR_K` has 8 non-positive values (0.077% of 10,412); `DSR_K` has 18 (0.088% of 20,496). `PTR_K` spans 10 to 1.26×10⁶ aside from those non-positive cells; `DSR_K` spans 10 to 2.79×10⁶. A susceptibility in SI cannot have a median near 3×10⁴, and a cgs susceptibility cannot either. 10⁻⁶ SI is the scale that lands in a physical range (median about 0.03 SI, maximum about 1–3 SI).

The package never defines `PTR`, `DSR`, `LUO`, `D`, `J`, or `K`. The drilling note lists laboratory petrophysical samples taken from core, and downhole susceptibility and gamma–gamma density, without mapping them onto these column names. See the questions below. `LUO_R` is a third series (finer spacing, values far outside the density range).

Median nearest-neighbour collar spacing, plan distance in KKJ metres: PTR holes 52.0 m (122 holes, none with `Z = 0`); DSR holes 48.1 m (141 holes, none with `Z = 0`).

### h) Invalid collars

522 unique `HOLE_ID`s, no embedded spaces. Plan coordinates all fall in the KKJ zone-3 bands. The elevation `Z = 0` set is 133 holes, identical to department `K` and `POS_ACCUR = 10`. Eight of them have 511P Cu (listed in §e). Horizontal positions for those eight exist; the elevation does not.

`POS_ACCUR = 99` is 26 holes, 20 of them with 511P Cu. The code is undefined in the documents read here, and those 26 have non-zero elevations inside the same bands.

21 plan positions are shared by two collars (42 holes). Among 511P holes, 20 positions are shared by two holes (40 holes), which is why the minimum nearest-neighbour distance is 0. Whether those are twin holes from one pad is not said in the tables.

## Counts against report §10

Nearest neighbour is the median, over holes, of the horizontal distance to the nearest other collar. For 511P it is 48.0 m including the eight `Z = 0` holes, and 48.0 m without them (281 holes).

| | Report §10 | These tables |
|---|---|---|
| Cu samples, single method 511P | 16,005 | 16,004 (`analy.txt`, `511P.txt`, and `d511p.dbf` agree; 0 deleted dBase rows) |
| Holes with Cu | 290 | 289, all present in the collar file and in `kalte.txt` |
| Cu below detection | 0.5% | 89 / 16,004 = 0.556% if both `<` and `!` count; 72 / 16,004 = 0.450% if only `<` counts |
| Median NN collar spacing | ~48 m | 48.0 m |
| Density / susceptibility | 10,412 core (122 holes) + 20,496 further (141 holes) | `PTR_*` 10,412 rows / 122 holes and `DSR_*` 20,496 rows / 141 holes |

The sample count and the hole count in §10 are each one above the source files. A merge of 511P with 511A would add 74 holes and 2,361 rows, not one hole and one row, so that merge does not explain the gap. The 0.5% figure is the same censoring rate at one significant figure; it is not an exact match to either flag definition. The 10,412 / 122 and 20,496 / 141 figures match `PTR_*` and `DSR_*` exactly. §10 calls the first set “core” and the second “further”. The column names do not say that.

Assay `KKJ_NORTH`, `KKJ_EAST`, and `Z` equal the collar on all 289 holes and do not change from one sample to the next. They are repeated collar coordinates. The extra pair `S_KKJ_NORT` / `S_KKJ_EAST` is also constant along the hole. It lies within 2 m of the collar on 267 holes, elsewhere in the KKJ bands on 21 holes, and outside those bands on 1 hole. It is not a desurveyed position.

## Confirmed for a later adapter

- CRS of the drillhole attributes: KKJ zone 3, EPSG:2393. Store easting = `KKJ_EAST` / `Kkj_y`, northing = `KKJ_NORTH` / `Kkj_X`. Do not read shapefile X as northing.
- Angles in degrees. Dip is positive downward, with +90 vertical.
- `Syvyys` is along-hole metres from the collar. `LENGTH` is the reported hole length and usually equals the deepest station.
- Cu for the sample table is 511P only, unit ppm. Censoring is the `CU_L` flag, not a negative value and not “every result ≤ 1 ppm”.
- Geophysical XYZ files use the same zone-3 magnitudes, with easting in the first coordinate column. That is the same CRS, not a second one.

## Questions before Stage 1

1. Azimuth: is `Suunta` / `AZIMUTH` clockwise from KKJ grid north? The drilling note does not say, and the many exact 0/90/180/270 values do not fix the sense of rotation.
2. Elevation datum of `Z`: the note describes how the collar was levelled and the years are 1984–1995, but it never names N60 or N2000.
3. The eight 511P holes with `Z = 0` listed in §e. Drop them, or is there an elevation source I should use?
4. Does `POS_ACCUR = 99` (26 collars, 20 with 511P) mean the collar is invalid?
5. `CU_L = "!"` (17 rows, including 9 zeros): treat it as below detection, same as `"<"`, or as something else?
6. `PTR_*` and `DSR_*`: please confirm both trios are density (`*_D`, kg/m³) and susceptibility (`*_K`, 10⁻⁶ SI), and which set is core versus a downhole log. The files and the drilling note do not define the prefixes. They are on disjoint holes, so they are not two measurements of one hole.
