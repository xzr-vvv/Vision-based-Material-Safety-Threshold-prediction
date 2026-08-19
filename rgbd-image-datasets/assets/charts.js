(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  var bar = echarts.init(document.getElementById('chart-bar'), null, { renderer: 'svg' });
  bar.setOption({
    animation: false,
    tooltip: { appendToBody: true, trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 150, right: 70, top: 20, bottom: 30 },
    xAxis: {
      type: 'value',
      axisLabel: { color: muted, formatter: function(v) { return (v / 1000) + 'K'; } },
      splitLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'category',
      data: ['TransCG', 'GraspNet-1Billion', 'YCB-Video', '华盛顿 RGB-D'],
      axisLabel: { color: ink, fontSize: 12 },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'bar',
      data: [
        { value: 57715, itemStyle: { color: muted } },
        { value: 97280, itemStyle: { color: accent } },
        { value: 133827, itemStyle: { color: accent } },
        { value: 250000, itemStyle: { color: accent } }
      ],
      barWidth: 22,
      itemStyle: { borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: 'right', color: ink, fontWeight: 700, formatter: function(p) { return p.value.toLocaleString() + ' 帧'; } }
    }]
  });

  window.addEventListener('resize', function() { bar.resize(); });
})();
