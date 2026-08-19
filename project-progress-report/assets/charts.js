(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  var el1 = document.getElementById('chart-accuracy');
  if (el1) {
    var c1 = echarts.init(el1, null, { renderer: 'svg' });
    c1.setOption({
      animation: false,
      tooltip: { trigger: 'axis', appendToBody: true, formatter: '{b}: {c}%' },
      grid: { left: 52, right: 28, top: 30, bottom: 56 },
      xAxis: {
        type: 'category',
        data: ['同分布验证集', '跨视角·整体', '跨视角·刚体', '跨视角·柔性', '跨视角·易碎'],
        axisLabel: { color: muted, fontSize: 12, interval: 0 },
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value', max: 100,
        axisLabel: { color: muted, fontSize: 12, formatter: '{value}%' },
        splitLine: { lineStyle: { color: rule } }
      },
      series: [{
        type: 'bar',
        barWidth: '46%',
        data: [
          { value: 88, itemStyle: { color: accent } },
          { value: 50, itemStyle: { color: accent2 } },
          { value: 93, itemStyle: { color: accent2 } },
          { value: 10, itemStyle: { color: accent2 } },
          { value: 0, itemStyle: { color: accent2 } }
        ],
        label: { show: true, position: 'top', color: ink, fontSize: 12, fontWeight: 600, formatter: '{c}%' }
      }]
    });
    window.addEventListener('resize', function () { c1.resize(); });
  }
})();
