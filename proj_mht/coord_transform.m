% --- 1. Define ECEF Coordinates (example values in meters) ---
% Using geodetic2ecef if you only have Lat/Lon/Alt
tx_ecef  = [-2691526; -4293641; 3857310]; % Transmitter
rx_ecef  = [-2690000; -4295000; 3858000]; % Receiver
tgt_ecef = [-2685000; -4290000; 3865000]; % Target

% --- 2. Calculate Distances (Slant Ranges) ---
R_T = norm(tgt_ecef - tx_ecef); % Range from Tx to Target
R_R = norm(tgt_ecef - rx_ecef); % Range from Rx to Target
L   = norm(rx_ecef  - tx_ecef); % Baseline (Tx to Rx)

bistatic_range = (R_T + R_R) - L;

% --- 3. Calculate Angle of Transmission (at Transmitter) ---
% We need the Tx geodetic coordinates to establish a local horizon
[tx_lat, tx_lon, tx_alt] = ecef2geodetic(wgs84Ellipsoid, ...
    tx_ecef(1), tx_ecef(2), tx_ecef(3));

% Transform Target ECEF to Azimuth/Elevation/Range relative to Tx
[az_tx, el_tx, ~] = ecef2aer(tgt_ecef(1), tgt_ecef(2), tgt_ecef(3), ...
    tx_lat, tx_lon, tx_alt, wgs84Ellipsoid);

% --- Display Results ---
fprintf('Bistatic Range: %.2f meters\n', bistatic_range);
fprintf('Transmission Azimuth: %.2f deg\n', az_tx);
fprintf('Transmission Elevation: %.2f deg\n', el_tx);
