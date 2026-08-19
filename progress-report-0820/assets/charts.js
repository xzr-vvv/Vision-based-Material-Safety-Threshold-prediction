(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  var baseAxis = {
    axisLine: { lineStyle: { color: rule } },
    axisLabel: { color: muted },
    splitLine: { lineStyle: { color: rule, type: 'dashed' } }
  };

  /* ---------- v1 族隔离训练曲线 ---------- */
  var v1Acc = [44.44,61.11,55.56,61.11,88.89,83.33,83.33,88.89,88.89,83.33,83.33,83.33,77.78,72.22,88.89,83.33,55.56,88.89,83.33,66.67,83.33,66.67,88.89,83.33,83.33,88.89,77.78,83.33,72.22,88.89,83.33,77.78,77.78,83.33,83.33,83.33,83.33,83.33,83.33,83.33,83.33];
  var v1Mae = [1.68,1.17,1.18,1.22,1.23,1.21,1.28,1.25,1.23,0.96,1.00,1.13,1.10,1.04,1.18,1.03,1.00,0.99,1.06,1.03,1.00,1.16,1.03,1.03,1.04,0.95,1.09,1.09,1.08,1.03,1.00,1.02,1.00,1.01,0.98,1.01,0.96,0.95,0.98,1.01,0.94];
  var v1Ep = v1Acc.map(function (_, i) { return i + 1; });

  var el1 = document.getElementById('chart-v1');
  if (el1) {
    var c1 = echarts.init(el1, null, { renderer: 'svg' });
    c1.setOption({
      animation: false,
      grid: { left: 52, right: 52, top: 46, bottom: 40 },
      legend: { data: ['验证准确率', 'F_min MAE'], textStyle: { color: muted }, top: 6 },
      tooltip: { trigger: 'axis', appendToBody: true },
      xAxis: Object.assign({ type: 'category', name: 'epoch', data: v1Ep, boundaryGap: false }, baseAxis),
      yAxis: [
        Object.assign({ type: 'value', name: '准确率 %', min: 40, max: 100 }, baseAxis),
        Object.assign({ type: 'value', name: 'MAE (N)', min: 0.8, max: 1.8, splitLine: { show: false }, axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } })
      ],
      series: [
        {
          name: '验证准确率', type: 'line', data: v1Acc, smooth: true, symbol: 'circle', symbolSize: 5,
          itemStyle: { color: accent }, lineStyle: { width: 2.5, color: accent },
          markLine: { symbol: 'none', silent: true, label: { formatter: '多数类基线 50%', color: muted, fontSize: 11 }, lineStyle: { color: muted, type: 'dashed' }, data: [{ yAxis: 50 }] }
        },
        {
          name: 'F_min MAE', type: 'line', yAxisIndex: 1, data: v1Mae, smooth: true, symbol: 'none',
          itemStyle: { color: accent2 }, lineStyle: { width: 2, color: accent2, type: 'solid' }
        }
      ]
    });
    window.addEventListener('resize', function () { c1.resize(); });
  }

  /* ---------- 深度流预训练曲线 ---------- */
  var ptEp = [10,20,30,40,50,60,70,80,90,100,110,120];
  var ptAcc = [82.0,82.9,84.8,85.8,87.8,88.1,87.6,90.4,91.4,91.7,91.2,91.1];
  var ptLoss = [1.039,0.913,0.882,0.853,0.798,0.790,0.766,0.740,0.718,0.714,0.734,0.716];

  var el2 = document.getElementById('chart-pretrain');
  if (el2) {
    var c2 = echarts.init(el2, null, { renderer: 'svg' });
    c2.setOption({
      animation: false,
      grid: { left: 52, right: 52, top: 46, bottom: 40 },
      legend: { data: ['视图对齐准确率', '对比损失 loss'], textStyle: { color: muted }, top: 6 },
      tooltip: { trigger: 'axis', appendToBody: true },
      xAxis: Object.assign({ type: 'category', name: 'epoch', data: ptEp, boundaryGap: false }, baseAxis),
      yAxis: [
        Object.assign({ type: 'value', name: '准确率 %', min: 75, max: 95 }, baseAxis),
        Object.assign({ type: 'value', name: 'loss', min: 0.6, max: 1.2, splitLine: { show: false }, axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } })
      ],
      series: [
        {
          name: '视图对齐准确率', type: 'line', data: ptAcc, smooth: true, symbol: 'circle', symbolSize: 6,
          itemStyle: { color: accent }, lineStyle: { width: 2.5, color: accent },
          markLine: { symbol: 'none', silent: true, label: { formatter: '随机基线 1.6%', color: muted, fontSize: 11 }, lineStyle: { color: muted, type: 'dashed' }, data: [{ yAxis: 76.5 }] }
        },
        {
          name: '对比损失 loss', type: 'line', yAxisIndex: 1, data: ptLoss, smooth: true, symbol: 'none',
          itemStyle: { color: accent2 }, lineStyle: { width: 2, color: accent2 }
        }
      ]
    });
    window.addEventListener('resize', function () { c2.resize(); });
  }
})();
