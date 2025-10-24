# Cron Job Setup - Automatic Daily Updates

## ✅ Cron Job Successfully Configured!

Your quantamental database will now update automatically every day at **6:00 PM**.

---

## 📋 Configuration Details

### **Schedule**
```
0 18 * * *  = Every day at 6:00 PM
```

### **Command**
```bash
cd /Users/andres/Downloads/Cursor/quant && \
/opt/anaconda3/envs/quant/bin/python scripts/update_daily.py >> logs/update.log 2>&1
```

### **What It Does**
1. Changes to project directory
2. Runs incremental update script
3. Logs output to `logs/update.log`
4. Captures both stdout and stderr

---

## 🔍 Monitoring Your Updates

### **View Latest Log**
```bash
tail -20 /Users/andres/Downloads/Cursor/quant/logs/update.log
```

### **Watch Updates in Real-Time**
```bash
tail -f /Users/andres/Downloads/Cursor/quant/logs/update.log
```
Press `Ctrl+C` to stop watching

### **View All Logs**
```bash
cat /Users/andres/Downloads/Cursor/quant/logs/update.log
```

### **Check Cron Job Status**
```bash
crontab -l
```

---

## 🧪 Verification

**Manual test completed successfully!** ✅

Output from test run:
```
📈 Updating prices... ✅
📊 Updating macro... ✅  
📉 Rebuilding factors... ✅
🦆 Updating DuckDB views... ✅
```

---

## 🛠️ Managing the Cron Job

### **Disable (Pause) Updates**
```bash
crontab -e
# Add # at the start of the line to comment it out:
# 0 18 * * * cd /Users/andres/...
```

### **Change Schedule**
```bash
crontab -e
```
Edit the timing:
- `0 18 * * 1` = Every Monday at 6 PM (weekly)
- `0 18 * * 1-5` = Weekdays only at 6 PM
- `0 2 1 * *` = First of month at 2 AM (monthly)

### **Remove Cron Job**
```bash
crontab -r  # Removes ALL cron jobs
```
Or edit and delete the line:
```bash
crontab -e
```

---

## 📊 Expected Behavior

### **When Data Is Up-to-Date**
```
📈 Prices are already up to date!
📊 Macro data is up to date
📉 Skipping factor rebuild (no new price data)
✅ Data is already up to date - no changes needed
```

### **When New Data Is Available**
```
📈 Added 5 new dates (2025-10-24 → 2025-10-29)
📊 New macro data available through 2025-10-29
📉 Rebuilt price factors: (12,862,584 rows)
✅ Incremental update completed successfully!
```

---

## 🚨 Troubleshooting

### **If Updates Don't Run**

1. **Check if cron is running:**
   ```bash
   ps aux | grep cron
   ```

2. **Check system logs:**
   ```bash
   grep CRON /var/log/system.log | tail -20
   ```

3. **Test command manually:**
   ```bash
   cd /Users/andres/Downloads/Cursor/quant && \
   /opt/anaconda3/envs/quant/bin/python scripts/update_daily.py
   ```

4. **Check permissions:**
   ```bash
   ls -la /Users/andres/Downloads/Cursor/quant/scripts/update_daily.py
   ```

### **Common Issues**

| Issue | Solution |
|-------|----------|
| No log file created | Check logs/ directory exists |
| Script not running | Verify Python path with `which python` in quant env |
| Permission denied | Run `chmod +x scripts/update_daily.py` |
| Script runs but fails | Check `logs/update.log` for error messages |

---

## 🎯 Next Steps

1. **Wait for first scheduled run** (today at 6 PM)
2. **Check logs after 6:05 PM:**
   ```bash
   cat /Users/andres/Downloads/Cursor/quant/logs/update.log
   ```
3. **Verify data updated:**
   ```bash
   conda activate quant
   python -c "from src.utils.io import get_last_date_from_parquet; from pathlib import Path; print(get_last_date_from_parquet(Path('data/factors/prices.parquet')))"
   ```

---

## 📅 Update Schedule Summary

| Event | Frequency | Action |
|-------|-----------|--------|
| **Automatic Update** | Daily at 6 PM | Cron runs `update_daily.py` |
| **Check Logs** | Weekly | Review `logs/update.log` |
| **Add New Stocks** | As needed | Run `python scripts/add_symbol.py TICKER` |
| **Full Backfill** | Quarterly | Run `python scripts/backfill_all.py` |

---

**Your quantamental research platform is now fully automated! 🚀**

For questions or modifications, see [QUICK_REFERENCE.md](QUICK_REFERENCE.md) or [INCREMENTAL_UPDATES.md](INCREMENTAL_UPDATES.md).

