"""
Servicio de procesamiento de texto
"""

from typing import List, Optional
from ..utils.file_parser import FileParser, split_text_into_chunks


class TextProcessor:
    """Procesador de texto"""
    
    @staticmethod
    def extract_from_files(file_paths: List[str]) -> str:
        """Extraer texto de multiples archivos"""
        return FileParser.extract_from_multiple(file_paths)
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Dividir texto en chunks
        
        Args:
            text: Texto original
            chunk_size: Tamano del chunk
            overlap: Tamano de superposicion
            
        Returns:
            Lista de chunks de texto
        """
        return split_text_into_chunks(text, chunk_size, overlap)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocesar texto
        - Eliminar espacios extra
        - Normalizar saltos de linea
        
        Args:
            text: Texto original
            
        Returns:
            Texto procesado
        """
        import re
        
        # Normalizar saltos de linea
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Eliminar lineas vacias consecutivas (maximo dos saltos de linea)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Eliminar espacios al inicio y final de cada linea
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """Obtener estadisticas del texto"""
        return {
            "total_chars": len(text),
            "total_lines": text.count('\n') + 1,
            "total_words": len(text.split()),
        }
