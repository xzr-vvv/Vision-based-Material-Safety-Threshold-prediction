(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // --- Chart 1: radar ---
  var radar = echarts.init(document.getElementById('chart-radar'), null, { renderer: 'svg' });
  radar.setOption({
    animation: false,
    tooltip: { appendToBody: true },
    legend: { bottom: 0, textStyle: { color: muted } },
    color: [accent, accent2, muted, accent + '88', accent2 + '88'],
    radar: {
      indicator: [
        { name: 'RGB-D 输入契合', max: 5 },
        { name: '力预测目标契合', max: 5 },
        { name: '数据/代码开放', max: 5 },
        { name: '三类物体覆盖', max: 5 },
        { name: '可直接复用度', max: 5 }
      ],
      axisName: { color: ink, fontSize: 12 },
      splitLine: { lineStyle: { color: rule } },
      splitArea: { areaStyle: { color: [bg2, '#f2f6fc'] } },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'radar',
      symbolSize: 4,
      data: [
        { value: [5, 4, 5, 2, 4], name: 'ForceSight' },
        { value: [5, 2, 2, 1, 2], name: '视觉力感知(RA-L)' },
        { value: [4, 3, 4, 2, 2], name: 'DeliGrasp' },
        { value: [2, 5, 2, 1, 2], name: 'V2F' },
        { value: [5, 1, 4, 1, 3], name: 'Hoi! 数据集' }
      ]
    }]
  });

  // --- Chart 2: bar ---
  var bar = echarts.init(document.getElementById('chart-bar'), null, { renderer: 'svg' });
  bar.setOption({
    animation: false,
    tooltip: { appendToBody: true, trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 130, right: 40, top: 20, bottom: 30 },
    xAxis: {
      type: 'value', max: 25,
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'category',
      data: ['Hoi! 数据集', 'V2F', '视觉力感知(RA-L)', 'DeliGrasp', 'ForceSight'],
      axisLabel: { color: ink, fontSize: 12 },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'bar',
      data: [14, 13, 13, 15, 20],
      barWidth: 22,
      itemStyle: { color: accent, borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: 'right', color: ink, fontWeight: 700 }
    }]
  });

  window.addEventListener('resize', function() {
    radar.resize();
    bar.resize();
  });
})();
