# 上传现有EmbryoFrameQC仓库

1. 解压`EmbryoFrameQC_GitHub_ready_20261113.zip`，得到`qc_rebuild_v1`文件夹。
2. 打开`https://github.com/zhao785164-sys/EmbryoFrameQC`的Code页。
3. 点击`Add file → Upload files`，把整个`qc_rebuild_v1`文件夹拖进去。
4. 检查文件路径以`qc_rebuild_v1/`开头。不要只拖里面零散文件到仓库根目录。
5. 提交说明可写：`Add rebuilt F0 QC workflow and reviewed audit records`。
6. 选择创建新分支，如`qc-rebuild-share`，提交并创建Pull Request。
7. 检查新增目录与README，合并到main后，将仓库的`qc_rebuild_v1/`目录链接交给师姐。

此包少于100个文件，单文件均低于25 MiB；符合本次核对的GitHub网页上传限制。
旧版README和scripts/frame_qc不会被同名覆盖。
归档内的原始93个文件作为一个压缩文件保留，常用代码和表格另行展开供浏览。

仓库当前为Public，上传的新内容也会公开。本包保留胚胎标识、图像文件名和原服务器路径以便复现；
不含原图，但这些研究记录仍应符合课题组的数据共享安排。
若只供组内使用，应使用私有共享仓库。模型/全量CSV可通过组内方式单独传输，不必放进Git历史。

来源：GitHub Docs，Adding a file to a repository（2026-11-13核对）。
https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
