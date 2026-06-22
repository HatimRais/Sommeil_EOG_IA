import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: false,
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
