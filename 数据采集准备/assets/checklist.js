(function () {
  var STORAGE_KEY = 'vtla-object-checklist-v1';
  var state = {};
  try {
    state = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
  } catch (e) { state = {}; }

  var boxes = document.querySelectorAll('input[type="checkbox"][data-id]');

  function save() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (e) {}
  }

  function updateCounters() {
    var totals = { FRA: [0, 0], SOF: [0, 0], RIG: [0, 0] };
    boxes.forEach(function (b) {
      if (b.checked) {
        totals[b.dataset.cat][0]++;
        if (b.dataset.star === '1') totals[b.dataset.cat][1]++;
      }
    });
    var all = 0, first = 0;
    ['FRA', 'SOF', 'RIG'].forEach(function (cat) {
      all += totals[cat][0];
      first += totals[cat][1];
      var done = document.getElementById('done-' + cat);
      var doneFirst = document.getElementById('done-first-' + cat);
      var count = document.getElementById('count-' + cat);
      var bar = document.getElementById('bar-' + cat);
      if (done) done.textContent = totals[cat][0];
      if (doneFirst) doneFirst.textContent = totals[cat][1];
      if (count) count.innerHTML = totals[cat][0] + '<small> / 60</small>';
      if (bar) bar.style.width = (totals[cat][0] / 60 * 100).toFixed(1) + '%';
    });
    var countAll = document.getElementById('count-all');
    var countFirst = document.getElementById('count-first');
    if (countAll) countAll.innerHTML = all + '<small> / 180</small>';
    if (countFirst) countFirst.innerHTML = first + '<small> / 90</small>';
  }

  boxes.forEach(function (b) {
    if (state[b.dataset.id]) b.checked = true;
    b.addEventListener('change', function () {
      if (b.checked) state[b.dataset.id] = 1; else delete state[b.dataset.id];
      save();
      updateCounters();
    });
  });

  var resetBtn = document.getElementById('btn-reset');
  if (resetBtn) {
    resetBtn.addEventListener('click', function () {
      if (confirm('确定清空全部勾选记录？此操作不可撤销。')) {
        state = {};
        save();
        boxes.forEach(function (b) { b.checked = false; });
        updateCounters();
      }
    });
  }

  updateCounters();
})();
