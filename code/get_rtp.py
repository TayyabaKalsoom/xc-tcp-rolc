import zipfile
z = zipfile.ZipFile(r"C:\Users\DELL\Downloads\rtp-torrent-v11.zip")
keep = [n for n in z.namelist()
        if (n.endswith(".csv") and n.count("/") >= 2
            and "/repo/" not in n and "sonarqube" not in n.lower())
        or n.endswith("readme.md")]
print(len(keep), "files")
for n in keep:
    z.extract(n, "rtp_csv")
print("done")