MATLAB UI for Map Tracks with Time SliderCreating a user interface in MATLAB involves designing the layout and implementing callback functions to handle user interactions. For a map with tracks and a time slider, you'll primarily use MATLAB's App Designer or build it programmatically using figure and uicontrol functions.1. UI Layout and ComponentsYou'll need the following core components for your UI:Figure: The main window for your application.Axes: An axes object where the map and tracks will be plotted.Slider (uicontrol with 'Style', 'slider'): To control the time displayed.Text Field (uicontrol with 'Style', 'text'): To display the current time selected by the slider.Push Buttons (uicontrol with 'Style', 'pushbutton'): For actions like "Load Track Data," "Clear Tracks," "Play/Pause Animation."Panel (uipanel): Optional, but useful for organizing UI elements (e.g., a panel for controls, and a panel for the map).Conceptual Code Structure (Programmatic Approach):% --- Main Figure ---
fig = figure('Name', 'Track Visualization', ...
             'NumberTitle', 'off', ...
             'Units', 'normalized', ...
             'Position', [0.1 0.1 0.8 0.8]); % Adjust position as needed

% --- Map Axes ---
ax = axes('Parent', fig, ...
          'Units', 'normalized', ...
          'Position', [0.05 0.25 0.9 0.7]); % Position for the map

% --- Controls Panel (Optional, but good practice) ---
controlPanel = uipanel('Parent', fig, ...
                       'Title', 'Controls', ...
                       'Units', 'normalized', ...
                       'Position', [0.05 0.05 0.9 0.15]);

% --- Time Slider ---
timeSlider = uicontrol('Parent', controlPanel, ...
                       'Style', 'slider', ...
                       'Units', 'normalized', ...
                       'Position', [0.05 0.5 0.7 0.4], ...
                       'Min', 0, 'Max', 100, 'Value', 0, ... % Min/Max/Value will be set dynamically
                       'Callback', @sliderCallback); % Assign callback function

% --- Current Time Display ---
timeText = uicontrol('Parent', controlPanel, ...
                     'Style', 'text', ...
                     'Units', 'normalized', ...
                     'Position', [0.8 0.5 0.15 0.4], ...
                     'String', 'Time: --:--:--');

% --- Load Data Button ---
loadButton = uicontrol('Parent', controlPanel, ...
                       'Style', 'pushbutton', ...
                       'Units', 'normalized', ...
                       'Position', [0.05 0.05 0.15 0.4], ...
                       'String', 'Load Data', ...
                       'Callback', @loadDataCallback);

% --- Clear Tracks Button ---
clearButton = uicontrol('Parent', controlPanel, ...
                        'Style', 'pushbutton', ...
                        'Units', 'normalized', ...
                        'Position', [0.25 0.05 0.15 0.4], ...
                        'String', 'Clear Tracks', ...
                        'Callback', @clearTracksCallback);

% --- Placeholder for global data or app data (e.g., in App Designer's properties) ---
% You'll need to store your track data (lat, lon, time) somewhere accessible
% to all callback functions.
appData.trackData = []; % Initialize empty
appData.mapHandle = []; % To store map object if using webmap
appData.plotHandle = []; % To store the handle of the plotted track

% Store handles in figure's UserData for programmatic access
set(fig, 'UserData', appData);

% --- Callback Functions (defined separately) ---
% function sliderCallback(src, event)
%     % Get current slider value
%     % Update timeText
%     % Redraw tracks based on selected time
% end
%
% function loadDataCallback(src, event)
%     % Open file dialog to select data file (e.g., CSV, MAT)
%     % Parse data (lat, lon, time)
%     % Update appData.trackData
%     % Set slider min/max based on time range
%     % Display initial track
% end
%
% function clearTracksCallback(src, event)
%     % Clear plotted lines from the axes
%     % Reset appData.trackData
%     % Reset slider
% end
2. Map DisplayYou have a few options for displaying a map:webmap (Recommended for online maps): This is the easiest way to display interactive online maps (OpenStreetMap, Esri, etc.).% Inside loadDataCallback or an initialization function
wm = webmap('OpenStreetMap'); % Or 'Esri Topographic', etc.
% You might need to set the view to encompass your track data
% wmzoom(wm, 'auto');
% wmcenter(wm, mean(trackData.latitude), mean(trackData.longitude));
When using webmap, it creates its own figure. You might need to integrate it more carefully if you want it within specific axes of your custom UI. A common approach is to create the webmap in a separate figure and then use axes in your UI to plot over it, or use uifigure and geoaxes in App Designer for better integration.Image Map (Offline): If you have a static map image (e.g., a satellite image or a scanned map) and know its geographic coordinates, you can display it using imshow and then geoshow or plot with appropriate coordinate transformations. This is more complex as it requires georeferencing the image.3. Track Data StructureYour track data should ideally be structured to easily access latitude, longitude, and time for each point. A table or a struct array is suitable:% Example data structure
trackData = table();
trackData.latitude = [34.0522, 34.0530, 34.0545, 34.0560];
trackData.longitude = [-118.2437, -118.2450, -118.2465, -118.2480];
trackData.time = datetime({'2025-07-23 10:00:00', '2025-07-23 10:01:00', ...
                           '2025-07-23 10:02:00', '2025-07-23 10:03:00'});
4. Drawing TracksYou'll use plot or geoplot (if using geoaxes in App Designer) to draw the tracks.Initial Plot: When data is loaded, plot the entire track.Dynamic Plotting (for slider): When the slider moves, you'll filter the trackData to include only points up to the selected time and then update the plotted line.% --- Inside loadDataCallback ---
% Assuming 'ax' is your axes handle and 'appData.trackData' is populated
% Set slider range based on time
minTime = min(appData.trackData.time);
maxTime = max(appData.trackData.time);
set(timeSlider, 'Min', datenum(minTime), 'Max', datenum(maxTime), 'Value', datenum(minTime));
set(timeSlider, 'SliderStep', [1/(numel(appData.trackData.time)-1) 10/(numel(appData.trackData.time)-1)]); % For discrete steps

% Plot initial track (e.g., only the first point or full track if desired)
% For webmap, you might use wmline
% For regular axes, use plot
hold(ax, 'on'); % Keep map visible when plotting
appData.plotHandle = plot(ax, appData.trackData.longitude(1), appData.trackData.latitude(1), ...
                          'r-o', 'LineWidth', 2, 'MarkerSize', 6);
hold(ax, 'off');
% Update appData in figure UserData
set(fig, 'UserData', appData);

% --- Inside sliderCallback ---
function sliderCallback(src, event)
    fig = get(src, 'Parent'); % Get the figure handle from the slider's parent
    if strcmp(get(fig, 'Type'), 'uipanel') % If slider is in a panel, get its parent figure
        fig = get(fig, 'Parent');
    end
    appData = get(fig, 'UserData'); % Retrieve appData

    currentTimeValue = get(src, 'Value');
    currentTime = datetime(currentTimeValue, 'ConvertFrom', 'datenum');
    set(appData.timeText, 'String', ['Time: ' datestr(currentTime, 'HH:MM:SS')]);

    % Filter data up to the current time
    idx = appData.trackData.time <= currentTime;
    currentLat = appData.trackData.latitude(idx);
    currentLon = appData.trackData.longitude(idx);

    % Update the plotted line
    if ~isempty(appData.plotHandle) && isvalid(appData.plotHandle)
        set(appData.plotHandle, 'XData', currentLon, 'YData', currentLat);
    else
        % Re-plot if handle is invalid (e.g., after clearing)
        hold(appData.ax, 'on');
        appData.plotHandle = plot(appData.ax, currentLon, currentLat, ...
                                  'r-o', 'LineWidth', 2, 'MarkerSize', 6);
        hold(appData.ax, 'off');
        set(fig, 'UserData', appData); % Save updated handle
    end
end
5. Time Slider FunctionalityThe sliderCallback function is crucial.It retrieves the slider's current value.Converts this value (which will be a datenum if you set Min/Max using datenum) back to a datetime object.Filters your trackData to include only points whose time is less than or equal to the slider's current time.Updates the XData and YData properties of the plotted line object.6. Interactivity and EnhancementsLoad Data: The loadDataCallback function should open a file selection dialog (uigetfile), read the data, parse latitude, longitude, and time, and then update the appData.trackData variable. It should also initialize the slider's Min, Max, and Value based on the loaded time range.Clear Tracks: The clearTracksCallback function should delete the plotted line object (delete(appData.plotHandle)) and reset appData.trackData and the slider.Play/Pause: You could add buttons and use a timer object to automatically increment the slider value over time, creating an animation.Zoom/Pan: If using webmap, this is built-in. For plot on regular axes, you'd use zoom on and pan on.Tooltips/Data Tips: Use datacursormode to show information about track points on hover.Using MATLAB App DesignerFor a more structured and visual approach, MATLAB App Designer is highly recommended. It allows you to drag-and-drop UI components and automatically generates the underlying code.Key advantages of App Designer:Visual Layout: Design your UI by dragging components.Code View: Easily switch between Design View and Code View.Callbacks: Right-click components to create callback functions.Properties: Manage app-specific data using properties (similar to appData in the programmatic example).geoaxes: App Designer provides a geoaxes component, which is specifically designed for plotting geographic data and integrates well with webmap functionality. This simplifies map display significantly.General Steps in App Designer:Open App Designer (appdesigner in the command window).Drag an Axes or Geo Axes component onto the canvas for your map.Drag a Slider and a Label for the time display.Drag Button components for "Load Data" and "Clear Tracks."In the Code View, define properties for trackData, mapHandle, plotHandle, etc.Implement the ValueChangingFcn (for continuous updates) or ValueChangedFcn (for updates when slider stops) callback for the slider.Implement ButtonPushedFcn callbacks for your buttons.This conceptual guide provides the necessary information to get started with building a MATLAB UI for track visualization. Remember to consult MATLAB's documentation for specific function details.
