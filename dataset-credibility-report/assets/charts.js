(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  var el = document.getElementById('chart-credibility');
  if (!el || typeof echarts === 'undefined') return;

  var chart = echarts.init(el, null, { renderer: 'svg' });
  chart.setOption({
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, appendToBody: true },
    grid: { left: 130, right: 60, top: 20, bottom: 40 },
    xAxis: {
      type: 'value',
      min: 0,
      max: 5,
      interval: 1,
      axisLabel: { color: muted, fontSize: 12 },
      splitLine: { lineStyle: { color: rule } },
      axisLine: { show: false }
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: ['Tabero', 'T-Rex', 'OopsieVerse', 'Deform360', 'RCT', 'SoftVTBench'],
      axisLabel: { color: ink, fontSize: 13, fontWeight: 600 },
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false }
    },
    series: [{
      type: 'bar',
      data: [5.0, 4.5, 4.2, 4.0, 3.5, 3.2],
      barWidth: '55%',
      itemStyle: {
        borderRadius: [0, 4, 4, 0],
        color: function(params) {
          var v = params.value;
          if (v >= 4.5) return accent;
          if (v >= 4.0) return accent2;
          return muted;
        }
      },
      label: {
        show: true,
        position: 'right',
        color: ink,
        fontSize: 12,
        fontWeight: 700,
        formatter: function(p) { return p.value.toFixed(1); }
      }
    }]
  });
  window.addEventListener('resize', function() { chart.resize(); });
})();
