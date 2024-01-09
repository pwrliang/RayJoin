#!/usr/bin/env bash

# 1       2  3        4    5          6             7           8         9            10          11             12        13                14            15               16
# ptsSize \t polySize fnId noPtPasses noConstraints executeTime ptMemTime ptRenderTime polyMemTime polyRenderTime setupTime triangulationTime polyIndexTime backendQueryTime accuracy

timing=$(tail -n 1 "$1")
pt_mem_time=$(echo "$timing" | cut -d$'\t' -f8,8)
pt_render_time=$(echo "$timing" | cut -d$'\t' -f9,9)
poly_mem_time=$(echo "$timing" | cut -d$'\t' -f10,10)
poly_render_time=$(echo "$timing" | cut -d$'\t' -f11,11)
triangle_time=$(echo "$timing" | cut -d$'\t' -f13,13)
echo "Preprocessing time $((triangle_time + pt_mem_time + poly_mem_time)) ms"
echo "Processing time $((pt_render_time + poly_render_time)) ms"