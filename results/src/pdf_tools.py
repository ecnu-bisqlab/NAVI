from pathlib import Path

def merge_pdfs_by_filename(input_dir: str | Path='.', output_pdf: str | Path='data_figs.pdf', pattern: str='*.pdf', recursive: bool=False):
    """按文件名排序合并 PDF。

    参数:
        input_dir: 待扫描 PDF 的目录
        output_pdf: 输出 PDF 路径
        pattern: 文件匹配模式，默认 *.pdf
        recursive: 是否递归子目录
    返回:
        (输出文件路径, 实际合并的文件列表)
    """
    from PyPDF2 import PdfMerger
    base = Path(input_dir)
    output_path = Path(output_pdf)
    candidates = base.rglob(pattern) if recursive else base.glob(pattern)
    pdfs = [p for p in candidates if p.is_file() and p.suffix.lower() == '.pdf']
    pdfs = sorted(pdfs, key=lambda p: p.name.lower())
    try:
        out_resolved = output_path.resolve()
        pdfs = [p for p in pdfs if p.resolve() != out_resolved]
    except FileNotFoundError:
        pass
    if not pdfs:
        raise ValueError(f'未找到可合并的 PDF: dir={base}, pattern={pattern}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merger = PdfMerger()
    try:
        for pdf in pdfs:
            merger.append(str(pdf))
        with open(output_path, 'wb') as f:
            merger.write(f)
    finally:
        merger.close()
    return (str(output_path), [str(p) for p in pdfs])

def merge_pdfs_in_order(pdf_files, output_pdf: str | Path='data_figs.pdf', base_dir: str | Path='.'):
    """按指定顺序合并 PDF。

    参数:
        pdf_files: 按目标顺序提供的 PDF 文件名/路径列表
        output_pdf: 输出 PDF 路径
        base_dir: 当 pdf_files 中是相对路径时的基准目录
    返回:
        (输出文件路径, 实际合并的文件列表)
    """
    if not pdf_files:
        raise ValueError('pdf_files 不能为空')
    from PyPDF2 import PdfMerger
    base = Path(base_dir)
    output_path = Path(output_pdf)
    ordered_paths = []
    for f in pdf_files:
        p = Path(f)
        if not p.is_absolute():
            p = base / p
        if not p.exists() or not p.is_file() or p.suffix.lower() != '.pdf':
            raise FileNotFoundError(f'PDF 不存在或非法: {p}')
        ordered_paths.append(p)
    try:
        out_resolved = output_path.resolve()
        ordered_paths = [p for p in ordered_paths if p.resolve() != out_resolved]
    except FileNotFoundError:
        pass
    if not ordered_paths:
        raise ValueError('指定的 PDF 列表在排除输出文件后为空')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merger = PdfMerger()
    try:
        for pdf in ordered_paths:
            merger.append(str(pdf))
        with open(output_path, 'wb') as f:
            merger.write(f)
    finally:
        merger.close()
    return (str(output_path), [str(p) for p in ordered_paths])

__all__ = ['merge_pdfs_by_filename', 'merge_pdfs_in_order']
