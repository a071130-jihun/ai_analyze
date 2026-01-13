from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass
class SleepStageAnnotation:
    stage_type: str
    start_time: float
    
    
class RMLParser:
    NAMESPACE = {"ps": "http://www.respironics.com/PatientStudy.xsd"}
    
    def __init__(self):
        self.stages: List[SleepStageAnnotation] = []
        self.recording_start: str = ""
        self.total_duration: float = 0.0
        
    def parse(self, rml_path: str) -> List[SleepStageAnnotation]:
        tree = ET.parse(rml_path)
        root = tree.getroot()
        
        self._parse_acquisition_info(root)
        self._parse_sleep_stages(root)
        
        return self.stages
    
    def _parse_acquisition_info(self, root: ET.Element):
        session = root.find(".//ps:Session", self.NAMESPACE)
        if session is not None:
            rec_start = session.find("ps:RecordingStart", self.NAMESPACE)
            duration = session.find("ps:Duration", self.NAMESPACE)
            
            if rec_start is not None and rec_start.text:
                self.recording_start = rec_start.text
            if duration is not None and duration.text:
                self.total_duration = float(duration.text)
    
    def _parse_sleep_stages(self, root: ET.Element):
        self.stages = []
        
        for stage in root.iter():
            if stage.tag.endswith("Stage"):
                stage_type = stage.get("Type")
                start_time = stage.get("Start")
                
                if stage_type and start_time:
                    self.stages.append(SleepStageAnnotation(
                        stage_type=stage_type,
                        start_time=float(start_time)
                    ))
        
        self.stages.sort(key=lambda x: x.start_time)
        return self.stages
    
    def get_epoch_labels(self, epoch_duration: int = 30) -> List[Tuple[float, float, str]]:
        if not self.stages:
            return []
        
        epoch_labels = []
        
        for i, stage in enumerate(self.stages):
            start = stage.start_time
            
            if i + 1 < len(self.stages):
                end = self.stages[i + 1].start_time
            elif self.total_duration > 0:
                end = self.total_duration
            else:
                end = start + epoch_duration
            
            epoch_labels.append((start, end, stage.stage_type))
        
        return epoch_labels
    
    def get_labels_at_intervals(
        self, 
        epoch_duration: int = 30,
        total_duration: float = 0.0
    ) -> List[str]:
        if total_duration <= 0:
            total_duration = self.total_duration
        if total_duration <= 0:
            return []
        
        num_epochs = int(total_duration / epoch_duration)
        labels = ["NotScored"] * num_epochs
        
        epoch_labels = self.get_epoch_labels(epoch_duration)
        
        for start, end, stage_type in epoch_labels:
            start_epoch = int(start / epoch_duration)
            end_epoch = int(end / epoch_duration)
            
            for epoch_idx in range(start_epoch, min(end_epoch, num_epochs)):
                labels[epoch_idx] = stage_type
        
        return labels


def find_rml_file(rml_dir: str, subject_id: str = None) -> str:
    rml_path = Path(rml_dir)
    
    if subject_id:
        rml_file = rml_path / f"{subject_id}.rml"
        if rml_file.exists():
            return str(rml_file)
        
        rml_files = list(rml_path.glob(f"{subject_id}*.rml"))
        if rml_files:
            return str(rml_files[0])
    
    rml_files = list(rml_path.glob("*.rml"))
    
    if not rml_files:
        raise FileNotFoundError(f"No RML file found in {rml_dir}")
    
    return str(rml_files[0])
