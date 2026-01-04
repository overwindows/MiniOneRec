# Feeds Recommendation Data Adaptation Guide

This guide explains how to adapt your Feeds recommendation data to work with MiniOneRec without modifying the core training code.

## Overview

MiniOneRec treats all items generically, so **feeds/articles/posts can be treated as "items"** just like Amazon products. The framework only requires the data to be in a specific format.

## Required Data Format

### 1. CSV Training Files (train/valid/test)

**Required columns:**
```csv
user_id,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid
```

**Mapping for Feeds:**
- `user_id` → User ID (e.g., "user_123")
- `history_item_title` → List of feed titles the user interacted with (chronologically ordered)
- `item_title` → Target feed title to predict
- `history_item_id` → List of feed IDs (numeric, e.g., `[456, 789]`)
- `item_id` → Target feed ID (numeric, e.g., `202`)
- `history_item_sid` → List of feed SIDs (e.g., `['<a_15><b_23><c_41>', '<a_8><b_12><c_67>']`)
- `item_sid` → Target feed SID (e.g., `'<a_20><b_5><c_33>'`)

**Example:**
```csv
user_id,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid
user_123,"['AI Breakthrough News', 'Tech Industry Update']","New ML Research Published",[456,789],202,"['<a_15><b_23><c_41>', '<a_8><b_12><c_67>']","<a_20><b_5><c_33>"
user_456,"['Sports Update', 'Weather Forecast']","Breaking News Alert",[101,202],303,"['<a_1><b_2><c_3>', '<a_4><b_5><c_6>']","<a_7><b_8><c_9>"
```

### 2. Item Metadata File (`feeds.item.json`)

**Format:**
```json
{
  "0": {
    "title": "Breaking: New AI Model Released",
    "description": "A new breakthrough in AI technology that promises to revolutionize...",
    "brand": "TechNews",  // Optional: publisher/source
    "categories": "Technology, AI, Machine Learning"  // Optional: feed categories/topics
  },
  "1": {
    "title": "Sports Update: Championship Results",
    "description": "Latest results from the championship match...",
    "brand": "SportsDaily",
    "categories": "Sports, Football"
  }
}
```

**Required fields:**
- `title`: Feed headline/title (required)
- `description`: Feed content summary/description (can be empty string)
- `brand`: Publisher/source (optional, can be empty string)
- `categories`: Topics/categories (optional, can be empty string)

### 3. SID Index File (`feeds.index.json`)

**Format:**
```json
{
  "0": ["<a_15>", "<b_23>", "<c_41>"],
  "1": ["<a_8>", "<b_12>", "<c_67>"],
  "2": ["<a_20>", "<b_5>", "<c_33>"]
}
```

**Structure:**
- Key: Item ID (string, e.g., "0", "1", "2")
- Value: Array of 3 SID tokens (3-level hierarchical codes)

## Feeds → Amazon Format Mapping

| Amazon Concept | Feeds Equivalent |
|----------------|------------------|
| Product | Feed/Article/Post |
| Purchase/Review | Click/Read/Like/Share/View |
| Product Title | Feed Title/Headline |
| Product Description | Feed Content/Summary |
| Product Category | Feed Category/Topic |
| Brand | Publisher/Source |

## Step-by-Step Adaptation Process

### Step 1: Prepare Sequential Interaction Data

You need user interaction sequences in chronological order:

**Your data format:**
```
User A: [Feed1, Feed2, Feed3] → Feed4 (to predict)
User A: [Feed1, Feed2, Feed3, Feed4] → Feed5 (to predict)
User B: [Feed10, Feed11] → Feed12 (to predict)
```

**Requirements:**
- Chronologically ordered interactions
- Each row represents: `user_history → next_feed`
- Users can have multiple rows (sliding window approach)

### Step 2: Generate SIDs for Feeds

You need to generate Semantic Item Descriptors (SIDs) for your feeds:

1. **Extract feed text:**
   - Concatenate: `title + description` (or `title + content`)
   - Similar to Amazon: `item_title + item_description`

2. **Generate embeddings:**
   ```bash
   bash rq/text2emb/amazon_text2emb.sh \
       --dataset Feeds \
       --root ./data/Feeds \
       --plm_name qwen \
       --plm_checkpoint your_qwen_model_path
   ```
   This creates: `Feeds.emb-qwen-td.npy`

3. **Train RQ-VAE/RQ-Kmeans:**
   ```bash
   # Option 1: RQ-VAE
   bash rq/rqvae.sh \
       --data_path ./data/Feeds/Feeds.emb-qwen-td.npy \
       --ckpt_dir ./output/Feeds \
       --lr 1e-3 \
       --epochs 10000 \
       --batch_size 20480
   
   # Option 2: RQ-Kmeans+
   bash rq/rqkmeans_plus.sh \
       --data_path ./data/Feeds/Feeds.emb-qwen-td.npy
   ```

4. **Generate SID index:**
   ```bash
   python rq/generate_indices.py
   # or
   bash rq/generate_indices_plus.sh
   ```
   This creates: `Feeds.index.json`

### Step 3: Create Item Metadata File

For each feed, extract:
- **Title**: Feed headline/title
- **Description**: Summary or content snippet (first 200-500 chars)
- **Categories**: Topics/tags (comma-separated)
- **Brand**: Publisher/source name

Save as: `Feeds.item.json` (format shown above)

### Step 4: Format CSV Files

Convert your interaction data to CSV format with all required columns:

**Python example:**
```python
import pandas as pd
import json

# Load your feeds data
feeds_data = [...]  # Your interaction sequences
feeds_metadata = {...}  # Your feed metadata
feeds_sids = {...}  # Your SID mappings

rows = []
for user_id, interactions in feeds_data.items():
    for i in range(1, len(interactions)):
        history = interactions[:i]
        target = interactions[i]
        
        # Get titles
        history_titles = [feeds_metadata[fid]['title'] for fid in history]
        target_title = feeds_metadata[target]['title']
        
        # Get IDs
        history_ids = history
        target_id = target
        
        # Get SIDs
        history_sids = [feeds_sids[str(fid)] for fid in history]
        target_sid = ''.join(feeds_sids[str(target)])  # Concatenate 3 tokens
        
        rows.append({
            'user_id': user_id,
            'history_item_title': history_titles,
            'item_title': target_title,
            'history_item_id': history_ids,
            'item_id': target_id,
            'history_item_sid': history_sids,
            'item_sid': target_sid
        })

df = pd.DataFrame(rows)
df.to_csv('Feeds_train.csv', index=False)
```

### Step 5: Organize Directory Structure

Create the following structure:
```
data/
  Feeds/
    train/
      Feeds_train.csv
    valid/
      Feeds_valid.csv
    test/
      Feeds_test.csv
    index/
      Feeds.index.json
      Feeds.item.json
      Feeds.emb-qwen-td.npy
    info/
      Feeds_info.txt  # Optional: SID -> Title -> ID mapping
```

## Code Changes Needed

### 1. Update `sft.sh`

```bash
# Change category name
for category in "Feeds"; do
    train_file=$(ls -f ./data/Feeds/train/${category}*.csv)
    eval_file=$(ls -f ./data/Feeds/valid/${category}*.csv)
    ...
    --category ${category} \
    --sid_index_path ./data/Feeds/index/Feeds.index.json \
    --item_meta_path ./data/Feeds/index/Feeds.item.json
done
```

### 2. Update `sft.py` (line ~74)

Add your category to the dictionary:
```python
category_dict = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Feeds": "news feeds and articles",  # Add this
    ...
}
```

### 3. Update `rl.sh` (if using RL)

Similar changes:
```bash
for category in "Feeds"; do
    ...
    --sid_index_path ./data/Feeds/index/Feeds.index.json \
    --item_meta_path ./data/Feeds/index/Feeds.item.json
done
```

## Special Considerations for Feeds

### 1. Temporal Aspects
- Feeds are time-sensitive
- Ensure chronological ordering in your sequences
- Consider recency in SID generation (recent feeds might need different SIDs)

### 2. Content Freshness
- Old feeds may become irrelevant
- Consider filtering by date range
- You might want separate models for different time periods

### 3. Multiple Interaction Types
- Feeds can have: click, read, like, share, comment
- **Option 1**: Use primary interaction type (e.g., "read")
- **Option 2**: Aggregate interactions (e.g., any interaction = positive)
- **Option 3**: Weight by interaction type

### 4. Feed Categories
- Use topics/categories as metadata
- Helps with semantic grouping in SID generation
- Can be used for filtering/constraints

### 5. Content Length
- Feed descriptions can be very long
- Consider truncating to first 200-500 characters
- Or use summary/abstract if available

## Example Conversion

**Your original feeds data:**
```
user_id: user_123
interactions: [
    {feed_id: 456, title: "AI Breakthrough", timestamp: "2024-01-01"},
    {feed_id: 789, title: "Tech Update", timestamp: "2024-01-02"},
    {feed_id: 101, title: "ML Research", timestamp: "2024-01-03"}
]
next_feed: {feed_id: 202, title: "New AI Model", timestamp: "2024-01-04"}
```

**Converted to MiniOneRec format:**
```csv
user_id,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid
user_123,"['AI Breakthrough', 'Tech Update', 'ML Research']","New AI Model",[456,789,101],202,"['<a_15><b_23><c_41>', '<a_8><b_12><c_67>', '<a_3><b_9><c_15>']","<a_20><b_5><c_33>"
```

## Validation Checklist

Before training, verify:

- [ ] CSV files have all 7 required columns
- [ ] `history_item_title` and `history_item_sid` are lists (use `eval()` format)
- [ ] `item_id` values match keys in `item.json`
- [ ] `item_sid` values match keys in `index.json`
- [ ] All item IDs in CSV exist in both `item.json` and `index.json`
- [ ] SIDs are 3-token format: `['<a_X>', '<b_Y>', '<c_Z>']`
- [ ] Sequences are chronologically ordered
- [ ] Category name updated in scripts
- [ ] File paths are correct

## Testing

After formatting your data:

1. **Quick validation:**
   ```python
   import pandas as pd
   import json
   
   # Load CSV
   df = pd.read_csv('Feeds_train.csv')
   print(f"Rows: {len(df)}")
   print(f"Columns: {df.columns.tolist()}")
   
   # Check first row
   row = df.iloc[0]
   print(f"History titles: {eval(row['history_item_title'])}")
   print(f"History SIDs: {eval(row['history_item_sid'])}")
   
   # Load JSON files
   with open('Feeds.index.json') as f:
       sids = json.load(f)
   print(f"SID entries: {len(sids)}")
   
   with open('Feeds.item.json') as f:
       items = json.load(f)
   print(f"Item entries: {len(items)}")
   ```

2. **Run training:**
   ```bash
   bash sft.sh
   ```

## Summary

✅ **Yes, feeds recommendation data can use the same format!**

The framework treats items generically, so feeds work as "items". You need to:

1. ✅ Generate SIDs for your feeds (using RQ-VAE pipeline)
2. ✅ Format your interaction data as CSV (7 columns)
3. ✅ Create `item.json` metadata file
4. ✅ Update category names in scripts (minimal changes)

The core training code works without modification once data is formatted correctly.

## Additional Resources

- See `convert_dataset.py` for reference conversion script
- See `data/amazon18_data_process.py` for data processing pipeline
- See `README.md` for full pipeline walkthrough

