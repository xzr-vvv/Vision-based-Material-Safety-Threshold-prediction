(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // --- Chart 1: bar ---
  var bar = echarts.init(document.getElementById('chart-bar'), null, { renderer: 'svg' });
  bar.setOption({
    animation: false,
    tooltip: { appendToBody: true, trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 170, right: 70, top: 20, bottom: 30 },
    xAxis: {
      type: 'value', min: 85, max: 91,
      axisLabel: { color: muted, formatter: '{value}%' },
      splitLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'category',
      data: ['DINOv2 ViT-g (上代)', 'DINOv3 (自监督)', 'SigLIP 2 (零样本)', 'PECore (零样本)', 'EVA-02-L (微调)', 'DINOv3 ViT-7B (IN-ReaL)'],
      axisLabel: { color: ink, fontSize: 12 },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'bar',
      data: [
        { value: 87.3, itemStyle: { color: muted } },
        { value: 88.4, itemStyle: { color: accent } },
        { value: 89.1, itemStyle: { color: accent } },
        { value: 89.3, itemStyle: { color: accent2 } },
        { value: 90.0, itemStyle: { color: accent2 } },
        { value: 90.4, itemStyle: { color: accent } }
      ],
      barWidth: 20,
      itemStyle: { borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: 'right', color: ink, fontWeight: 700, formatter: '{c}%' }
    }]
  });

  // --- Chart 2: radar ---
  var radar = echarts.init(document.getElementById('chart-radar'), null, { renderer: 'svg' });
  radar.setOption({
    animation: false,
    tooltip: { appendToBody: true },
    legend: { bottom: 0, textStyle: { color: muted } },
    color: [accent, accent2, muted],
    radar: {
      indicator: [
        { name: '识别精度潜力', max: 5 },
        { name: '密集/几何特征', max: 5 },
        { name: '开放与易用性', max: 5 },
        { name: '多模态能力', max: 5 },
        { name: '时效性', max: 5 }
      ],
      radius: '62%',
      axisName: { color: ink, fontSize: 12 },
      splitLine: { lineStyle: { color: rule } },
      splitArea: { areaStyle: { color: [bg2, '#f6f2fc'] } },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'radar',
      symbolSize: 4,
      data: [
        { value: [5, 5, 4, 2, 5], name: 'DINOv3 (RGB 流)' },
        { value: [5, 3, 4, 5, 5], name: 'SigLIP 2 (语义)' },
        { value: [4, 5, 4, 5, 5], name: 'Depth Anything V2 (深度流)' }
      ]
    }]
  });

  window.addEventListener('resize', function() {
    bar.resize();
    radar.resize();
  });
})();
