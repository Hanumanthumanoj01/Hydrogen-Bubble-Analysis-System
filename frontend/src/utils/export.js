/**
 * export.js — Excel / CSV export utilities
 * Uses the SheetJS (xlsx) library
 */
import * as XLSX from 'xlsx'

/**
 * Export data array to .xlsx file.
 * @param {Array}  rows      - array of plain objects
 * @param {Array}  cols      - column keys (defines column order)
 * @param {string} filename  - e.g. 'bubble_analysis.xlsx'
 * @param {Object} extras    - optional { sheet2: { name, data, cols }, ... }
 */
export function exportToExcel(rows, cols, filename = 'export.xlsx', extras = {}) {
  const wb = XLSX.utils.book_new()

  // Main sheet
  const ws = XLSX.utils.json_to_sheet(rows, { header: cols })
  autoWidth(ws, rows, cols)
  XLSX.utils.book_append_sheet(wb, ws, 'Data')

  // Extra sheets (e.g. summary)
  for (const key of Object.keys(extras)) {
    const { name, data, cols: eCols } = extras[key]
    const ws2 = XLSX.utils.json_to_sheet(data, { header: eCols })
    autoWidth(ws2, data, eCols)
    XLSX.utils.book_append_sheet(wb, ws2, name || key)
  }

  XLSX.writeFile(wb, filename)
}

/**
 * Export a simple array of objects to a .csv download.
 */
export function exportToCSV(rows, cols, filename = 'export.csv') {
  const header = cols.join(',')
  const body = rows.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(',')).join('\n')
  const blob = new Blob([header + '\n' + body], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

function autoWidth(ws, rows, cols) {
  const widths = cols.map(c => Math.max(
    String(c).length,
    ...rows.map(r => String(r[c] ?? '').length)
  ) + 2)
  ws['!cols'] = widths.map(w => ({ wch: Math.min(w, 40) }))
}
