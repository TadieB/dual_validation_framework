// GEE SCRIPT WITH A SAFETY BUFFER ADDED TO THE ROI

// Step 1: Define the ROI with a buffer.
// We add a small buffer (e.g., 0.25 degrees) to ensure full coverage.
var buffer_degrees = 0.25;
var roi_buffered = ee.Geometry.Rectangle([
  -0.5 - buffer_degrees, // minLon
  40.0 - buffer_degrees, // minLat
  28.5 + buffer_degrees, // maxLon
  60.0 + buffer_degrees  // maxLat
]);

// Step 2: Load the Copernicus World Cover 2021 image.
var worldCover = ee.Image('ESA/WorldCover/v200/2021').select('Map');

// Step 3: Clip the global image to your buffered ROI.
var worldCover_roi = worldCover.clip(roi_buffered);

// Step 4: Center the map on your ROI to visualize it (optional).
Map.centerObject(roi_buffered, 5);
Map.addLayer(worldCover_roi, {}, 'World Cover Clipped to Buffered ROI');

// // Step 5: Export the final, clipped image to your Google Drive.
// Export.image.toDrive({
//   image: worldCover_roi,
//   description: 'WorldCover_for_MODIS_ROI_Buffered', // New file name
//   folder: 'GEE_Exports',
//   scale: 10,
//   region: roi_buffered,
//   maxPixels: 1e11 // Increased maxPixels slightly for the larger area
// });


// // GEE SCRIPT WITH ACCURATE BOUNDING BOX FOR THE 4 MODIS TILES

// // Step 1: Define the ROI using the accurate extent of the tile block.
// // This covers the full width of the sinusoidal tiles h18 and h19 at this latitude.
// var roi = ee.Geometry.Rectangle([
//   -0.5, // Minimum Longitude (covers western edge of h18v04)
//   40,   // Minimum Latitude (40°N)
//   28.5, // Maximum Longitude (covers eastern edge of h19v04)
//   60    // Maximum Latitude (60°N)
// ]);

// // Step 2: Load the Copernicus World Cover 2021 image.
// var worldCover = ee.Image('ESA/WorldCover/v200/2021').select('Map');

// // Step 3: Clip the global image to your ROI.
// var worldCover_roi = worldCover.clip(roi);

// // Step 4: Center the map on your ROI to visualize it (optional).
// Map.centerObject(roi, 5);
// Map.addLayer(worldCover_roi, {}, 'World Cover Clipped to MODIS Footprint');

// // Step 5: Export the final, clipped image to your Google Drive.
// Export.image.toDrive({
//   image: worldCover_roi,
//   description: 'WorldCover_for_MODIS_ROI_Exact', // New file name
//   folder: 'GEE_Exports',
//   scale: 10,
//   region: roi,
//   maxPixels: 1e10
// });