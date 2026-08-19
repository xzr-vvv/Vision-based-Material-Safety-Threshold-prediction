(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  var DATA = {"刚体": [[11.0, 0.25], [202.0, 1.5], [368.0, 2.5], [193.0, 2.0], [105.0, 1.5], [309.0, 5.0], [154.0, 2.0], [489.0, 1.75], [385.0, 2.5], [216.0, 2.0], [201.0, 2.0], [10.0, 0.25], [88.0, 1.0], [181.0, 1.0], [373.0, 2.0], [370.0, 2.0], [134.0, 3.0], [338.0, 2.0], [130.0, 2.0], [91.0, 1.0], [51.0, 1.0], [33.0, 2.0], [36.0, 5.0], [508.0, 1.75], [150.0, 0.75], [279.0, 1.0], [99.0, 1.25], [127.0, 2.0], [286.0, 3.0], [112.0, 1.25], [370.0, 2.0], [612.0, 2.0], [236.0, 2.0], [395.0, 2.25], [439.0, 3.25], [423.0, 2.5], [349.0, 2.5], [394.0, 1.5], [416.0, 2.0], [794.0, 4.5], [511.0, 2.0], [175.0, 1.5], [123.0, 1.0], [413.0, 3.0], [801.0, 3.25], [733.0, 5.0], [431.0, 2.0], [337.0, 2.0], [89.0, 1.0], [961.0, 3.5], [641.0, 2.0], [133.0, 1.5], [2.0, 0.25], [83.0, 1.0], [869.0, 3.5], [70.0, 1.5], [30.0, 1.0], [280.0, 1.75], [305.0, 2.5], [225.0, 1.5], [1384.0, 7.0]], "柔性": [[1.0, 0.25], [2.0, 0.25], [11.0, 0.25], [10.0, 0.25], [42.0, 0.5], [27.0, 0.5], [6.0, 0.25], [89.0, 2.0], [79.0, 2.0], [396.0, 2.0], [105.0, 0.5], [297.0, 1.5], [45.0, 1.0], [41.0, 0.5], [162.0, 1.5], [168.0, 1.5], [237.0, 1.5], [96.0, 1.0], [74.0, 1.0], [7.0, 0.5], [9.0, 0.5], [1.0, 0.25], [1.0, 0.25], [4.0, 0.25], [7.0, 0.25], [4.0, 0.25], [6.0, 0.25], [206.0, 1.0], [237.0, 1.5], [181.0, 0.75], [238.0, 1.5], [231.0, 1.5], [282.0, 1.5], [387.0, 2.0], [23.0, 0.5], [17.0, 0.5], [15.0, 0.5], [40.0, 0.5], [64.0, 1.0], [88.0, 1.0], [8.0, 0.5], [11.0, 0.5], [15.0, 0.5], [8.0, 0.5], [18.0, 0.5], [11.0, 0.5], [13.0, 0.5], [10.0, 0.5], [11.0, 0.5], [11.0, 0.5], [1.0, 0.25], [4.0, 0.25], [5.0, 0.25], [220.0, 2.0], [273.0, 2.0], [14.0, 0.25], [21.0, 0.25]], "易碎": [[145.0, 0.75], [2.0, 0.25], [278.0, 1.5], [64.0, 0.5], [62.0, 0.5], [635.0, 2.5], [286.0, 1.5], [274.0, 1.0], [204.0, 1.0], [245.0, 1.0], [424.0, 1.5]]};
  var COLORS = {'刚体': accent, '柔性': '#0f7a4d', '易碎': accent2};

  function curve(mu, color) {
    var pts = [];
    for (var m = 0; m <= 500; m += 10) {
      pts.push([m, m / 1000 * 9.81 / (2 * mu)]);
    }
    return { type: 'line', data: pts, showSymbol: false, silent: true,
      lineStyle: { color: color, width: 2, type: 'dashed', opacity: 0.7 }, tooltip: { show: false } };
  }

  var scatter = echarts.init(document.getElementById('chart-scatter'), null, { renderer: 'svg' });
  scatter.setOption({
    animation: false,
    tooltip: { appendToBody: true, formatter: function(p) { return p.seriesName + '<br>质量: ' + p.value[0] + ' g<br>最小抓力: ' + p.value[1] + ' N'; } },
    legend: { bottom: 0, textStyle: { color: muted } },
    grid: { left: 60, right: 30, top: 20, bottom: 70 },
    xAxis: { name: '质量 (g)', nameLocation: 'middle', nameGap: 30, type: 'value', axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    yAxis: { name: '最小抓力 (N)', type: 'value', axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: '刚体', type: 'scatter', data: DATA['刚体'], symbolSize: 7, itemStyle: { color: COLORS['刚体'], opacity: 0.75 } },
      { name: '柔性', type: 'scatter', data: DATA['柔性'], symbolSize: 7, itemStyle: { color: COLORS['柔性'], opacity: 0.75 } },
      { name: '易碎', type: 'scatter', data: DATA['易碎'], symbolSize: 9, itemStyle: { color: COLORS['易碎'], opacity: 0.85 } },
      curve(0.66, COLORS['刚体']), curve(0.22, COLORS['柔性']), curve(0.95, COLORS['易碎'])
    ]
  });

  var mae = echarts.init(document.getElementById('chart-mae'), null, { renderer: 'svg' });
  mae.setOption({
    animation: false,
    tooltip: { appendToBody: true, trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { bottom: 0, textStyle: { color: muted } },
    grid: { left: 60, right: 30, top: 20, bottom: 70 },
    xAxis: { type: 'category', data: ['刚体', '柔性', '易碎'], axisLabel: { color: ink, fontWeight: 700 } },
    yAxis: { name: 'MAE (N)', type: 'value', axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: '物理公式 F=mg/2μ', type: 'bar', data: [0.99, 1.24, 0.26], barWidth: 36,
        itemStyle: { color: accent2, borderRadius: [4,4,0,0] },
        label: { show: true, position: 'top', color: ink, fontWeight: 700, formatter: '{c}N' },
        markLine: { symbol: 'none', silent: true, lineStyle: { color: accent, type: 'dashed', width: 2 },
          label: { color: accent, formatter: 'CNN 整体 MAE = 0.60N' },
          data: [{ yAxis: 0.60 }] } }
    ]
  });

  window.addEventListener('resize', function() { scatter.resize(); mae.resize(); });
})();
