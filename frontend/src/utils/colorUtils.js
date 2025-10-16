export function getContrastTextColor(bgColor) {
  // Accepts Bootstrap color names or hex codes
  const colorMap = {
    primary: "#0d6efd",
    secondary: "#6c757d",
    success: "#198754",
    danger: "#dc3545",
    warning: "#ffc107",
    info: "#0dcaf0",
    light: "#f8f9fa",
    dark: "#212529"
  };
  let color = colorMap[bgColor] || bgColor;
  if (color.startsWith("#")) {
    // Convert hex to RGB
    const r = parseInt(color.substr(1, 2), 16);
    const g = parseInt(color.substr(3, 2), 16);
    const b = parseInt(color.substr(5, 2), 16);
    // Calculate brightness
    const brightness = (r * 299 + g * 587 + b * 114) / 1000;
    return brightness > 128 ? "#212529" : "#fff"; // dark text for light bg, white for dark bg
  }
  // Default to white text for unknown colors
  return "#fff";
}