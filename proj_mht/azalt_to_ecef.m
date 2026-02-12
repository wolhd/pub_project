function X_ecef = bistatic_az_alt_to_ecef(T_ecef, R_ecef, az_deg, h_target, rho_b)
%--------------------------------------------------------------------------
% BISTATIC_AZ_ALT_TO_ECEF
%
% Computes target ECEF position given:
%   - Transmitter ECEF position (3x1) [m]
%   - Receiver ECEF position (3x1) [m]
%   - Transmit azimuth (deg, ENU frame at transmitter)
%   - Target altitude above WGS84 ellipsoid (m)
%   - Bistatic range (|X-T| + |X-R|) (m)
%
% Unknown elevation is solved via 1D root finding.
%
% OUTPUT:
%   X_ecef : 3x1 ECEF position (meters)
%
% Author: ChatGPT
%--------------------------------------------------------------------------

%% --- WGS84 constants ---
a  = 6378137.0;                 % semi-major axis
f  = 1/298.257223563;           
e2 = f*(2-f);                   % eccentricity squared

%% --- Convert transmitter to geodetic ---
[latT, lonT, ~] = ecef2geodetic_wgs84(T_ecef, a, e2);

%% --- Build ENU->ECEF rotation matrix ---
R_ENU2ECEF = enu2ecef_matrix(latT, lonT);

az = deg2rad(az_deg);

%% --- Root solve for elevation ---
% Elevation search bracket (adjust if needed)
el_low  = deg2rad(-5);
el_high = deg2rad(60);

residual_fun = @(el) bistatic_residual( ...
    el, T_ecef, R_ecef, az, h_target, rho_b, ...
    R_ENU2ECEF, a, e2);

% Solve
el_sol = fzero(residual_fun, [el_low el_high]);

%% --- Construct final LOS vector ---
u_enu = [cos(el_sol)*sin(az);
         cos(el_sol)*cos(az);
         sin(el_sol)];

u_ecef = R_ENU2ECEF * u_enu;

%% --- Solve for altitude intersection ---
s = solve_altitude_intersection(T_ecef, u_ecef, h_target, a, e2);

%% --- Final ECEF position ---
X_ecef = T_ecef + s*u_ecef;

end

%==========================================================================
%========================= Helper Functions ===============================
%==========================================================================

function F = bistatic_residual(el, T, R, az, h_target, rho_b, Rmat, a, e2)
% Computes bistatic residual for a given elevation

u_enu = [cos(el)*sin(az);
         cos(el)*cos(az);
         sin(el)];

u = Rmat * u_enu;

% Intersect ray with altitude surface
s = solve_altitude_intersection(T, u, h_target, a, e2);

if isnan(s)
    F = 1e6;
    return
end

X = T + s*u;

rho = norm(X - T) + norm(X - R);

F = rho - rho_b;

end

%--------------------------------------------------------------------------

function s = solve_altitude_intersection(T, u, h_target, a, e2)
% Solves for intersection of ray with constant-altitude surface
% Uses 1D root finding in range parameter s

height_fun = @(s) height_error(T + s*u, h_target, a, e2);

% Initial guess assuming flat Earth vertical approx
if abs(u(3)) < 1e-6
    s = NaN;
    return
end

s0 = h_target / u(3);

try
    s = fzero(height_fun, s0);
catch
    s = NaN;
end

end

%--------------------------------------------------------------------------

function err = height_error(X, h_target, a, e2)
% Returns difference between point height and desired height

[~,~,h] = ecef2geodetic_wgs84(X, a, e2);
err = h - h_target;

end

%--------------------------------------------------------------------------

function R = enu2ecef_matrix(lat, lon)
% Builds ENU to ECEF rotation matrix

slat = sin(lat); clat = cos(lat);
slon = sin(lon); clon = cos(lon);

R = [-slon,            clon,           0;
     -slat*clon,      -slat*slon,     clat;
      clat*clon,       clat*slon,     slat];
end

%--------------------------------------------------------------------------

function [lat, lon, h] = ecef2geodetic_wgs84(X, a, e2)
% Converts ECEF to geodetic (iterative Newton method)

x = X(1); y = X(2); z = X(3);

lon = atan2(y,x);
r   = sqrt(x^2 + y^2);

lat = atan2(z, r*(1-e2));

for k = 1:6
    N = a / sqrt(1 - e2*sin(lat)^2);

    h = r/cos(lat) - N;
    lat = atan2(z, r*(1 - e2*N/(N+h)));
end

N = a / sqrt(1 - e2*sin(lat)^2);
h = r/cos(lat) - N;

end
