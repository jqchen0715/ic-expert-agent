import re
from langchain_text_splitters import TextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class ICCustomTextSplitter(TextSplitter):
    """针对集成电路(IC)文档的定制化分割器：优先切分Verilog代码块、时序图、章节边界"""

    def __init__(
            self,
            chunk_size: int = 800,
            chunk_overlap: int = 100,
            separators: list[str] | None = None,
    ):
        # 关键：调用父类初始化，基类会把chunk_size存为self._chunk_size
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or [
            # 1. 优先切分Verilog代码块边界
            r"(?<=\endmodule)",  # module结尾后切分（正向后顾，保留endmodule）
            r"(?=module\s+)",  # module开头前切分（正向前瞻，保留module）
            r"```verilog[\s\S]*?```",  # 匹配```verilog```包裹的代码块
            # 2. 时序图边界（匹配“图X 时序图/波形图/Timing Diagram”等）
            r"(?<=\n)(图\d+[\u4e00-\u9fa5]*?(时序图|波形图|Timing Diagram))(?=\n)",
            # 3. 章节边界（匹配 1.1、2.3.4 等章节号）
            r"(?<=\n)(\d+(\.\d+)+[\u4e00-\u9fa5\s]*?[:：])",
            # 4. 通用分隔符（兜底）
            "\n\n", "\n", ". ", " ", ""
        ]
        # 细切分器（粗切分后超长时使用）
        self.fine_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,  # 改：self.chunk_size → self._chunk_size
            chunk_overlap=self._chunk_overlap,  # 改：self.chunk_overlap → self._chunk_overlap
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(self, text: str) -> list[str]:
        """核心切分逻辑：先粗切，再细切"""
        # 步骤1：按Verilog代码块粗切
        verilog_blocks = self._split_verilog_blocks(text)
        # 步骤2：对非Verilog块，按时序图、章节粗切
        final_chunks = []
        for block in verilog_blocks:
            if self._is_verilog_block(block):  # Verilog块直接保留
                final_chunks.append(block)
            else:  # 非Verilog块：按时序图、章节切分
                chapter_timing_chunks = self._split_chapter_timing(block)
                # 步骤3：对超长块做细切分
                for chunk in chapter_timing_chunks:
                    # 关键修复：self.chunk_size → self._chunk_size
                    if len(chunk) > self._chunk_size:
                        fine_chunks = self.fine_splitter.split_text(chunk)
                        final_chunks.extend(fine_chunks)
                    else:
                        final_chunks.append(chunk)
        return final_chunks

    def _split_verilog_blocks(self, text: str) -> list[str]:
        """切分Verilog代码块（module/endmodule、```verilog```）"""
        # 匹配module/endmodule包裹的代码块
        verilog_pattern = r"(module\s+[\w_]+\s*[\s\S]*?endmodule)"
        parts = re.split(verilog_pattern, text, flags=re.IGNORECASE)
        # 过滤空字符串，重组代码块（保留完整module）
        result = []
        for i in range(len(parts)):
            if parts[i].strip() == "":
                continue
            # 如果是module块（奇数位，因为split后是[非代码, 代码, 非代码, 代码...]）
            if i % 2 == 1:
                result.append(parts[i].strip())
            else:
                result.append(parts[i].strip())
        return result

    def _is_verilog_block(self, text: str) -> bool:
        """判断是否为Verilog代码块"""
        return re.search(r"module\s+[\w_]+\s*?;|endmodule", text, flags=re.IGNORECASE) is not None

    def _split_chapter_timing(self, text: str) -> list[str]:
        """按章节、时序图切分非Verilog文本"""
        # 章节正则：匹配 1.1、2.3.4 等章节标题（如“1.1 集成电路基础”）
        chapter_pattern = r"(?<=\n)(\d+(\.\d+)+[\u4e00-\u9fa5\s]*?[:：])"
        # 时序图正则：匹配“图X 时序图/波形图”
        timing_pattern = r"(?<=\n)(图\d+[\u4e00-\u9fa5]*?(时序图|波形图|Timing Diagram))(?=\n)"

        # 先按时序图切分
        timing_chunks = re.split(timing_pattern, text)
        timing_chunks = [c.strip() for c in timing_chunks if c.strip()]

        # 再按章节切分
        final_chunks = []
        for chunk in timing_chunks:
            chapter_chunks = re.split(chapter_pattern, chunk)
            chapter_chunks = [c.strip() for c in chapter_chunks if c.strip()]
            final_chunks.extend(chapter_chunks)
        return final_chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """适配Document对象的切分（保留元数据如页码）"""
        split_docs = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for chunk in text_chunks:
                split_docs.append(Document(
                    page_content=chunk,
                    metadata=doc.metadata  # 保留原文档的页码、来源等元数据
                ))
        return split_docs