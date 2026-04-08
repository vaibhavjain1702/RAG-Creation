# 🚀 How to Run on Google Colab (Free GPU)

Running the full experiment on your Mac takes ~4 hours because Phi-2 has to run on the CPU to avoid numerical glitches. 

By running this on **Google Colab**, the script will automatically detect the **NVIDIA T4 GPU** (via the code changes we just made to `generator.py`) and use `cuda` with `float16`. 
**This will drop the 4-hour runtime down to just 10-15 minutes.**

Follow these exact steps:

### Step 1: Prepare your folder
1. On your Mac, right-click the `Assignment 1` folder and click **Compress "Assignment 1"**.
2. This will create an `Assignment 1.zip` file.

### Step 2: Set up Colab
1. Go to [Google Colab](https://colab.research.google.com/) and sign in.
2. Click **New Notebook**.
3. In the menu bar at the top, click **Runtime** > **Change runtime type**.
4. Select **T4 GPU** as the hardware accelerator and click Save.

### Step 3: Upload and Run
Click the **Folder icon** 📁 on the left sidebar in Colab to open the file browser. 
Drag and drop your `Assignment 1.zip` file into that sidebar to upload it.

Create a new code cell in the notebook, paste the following code, and press **Play**:

```bash
# 1. Unzip the folder
!unzip -q "Assignment 1.zip" -d /content/
%cd /content/Assignment 1

# 2. Install requirements
!pip install -r requirements.txt

# 3. Run the full experiment! (Takes ~10 minutes on GPU)
!python main.py --mode full
```

### Step 4: Download your Results
1. Once it finishes running, you will see it say `Results saved to ...summary_results.csv`.
2. In the left folder sidebar, go into `/content/Assignment 1/results/`.
3. Right-click the `summary_results.csv` and `evaluation_results.csv` files and click **Download**.
4. Replace the old files on your Mac's `results` folder with these new ones.

### Step 5: View the Magic
Back on your Mac, simply run:
```bash
streamlit run app.py
```
Go to the **Comparison Dashboard** tab, and you'll see all your GPU-generated data beautifully visualized to show your teacher!
