"""
Vertex AI Fine-tuning Pipeline for AfriMed CHW Assistant

This module handles:
1. Data preparation and upload to GCS
2. Fine-tuning job creation and monitoring
3. Model deployment for inference
"""

import os
import json
import yaml
import click
from pathlib import Path
from typing import Optional
from datetime import datetime

import vertexai
from vertexai.tuning import sft
from vertexai.generative_models import GenerativeModel
from google.cloud import storage
import structlog

logger = structlog.get_logger()


class AfriMedFineTuner:
    """Handles Gemini fine-tuning for AfriMed CHW Assistant."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        self.config = self._load_config(config_path)
        self.project_id = os.environ.get("GCP_PROJECT_ID", self.config["gcp"]["project_id"])
        self.region = self.config["gcp"]["region"]
        self.staging_bucket = self.config["gcp"]["staging_bucket"].replace("${GCP_PROJECT_ID}", self.project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.region)
        logger.info("Initialized Vertex AI", project=self.project_id, region=self.region)
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def prepare_training_data(self, input_file: str, output_file: str) -> str:
        """
        Convert training data to Vertex AI format and upload to GCS.
        
        Input format: JSONL with messages array
        Output format: Vertex AI supervised tuning format
        """
        logger.info("Preparing training data", input_file=input_file)
        
        processed_examples = []
        system_prompt = self.config.get("system_prompt", "")
        
        with open(input_file, "r") as f:
            for line in f:
                example = json.loads(line)
                
                # Convert to Vertex AI format
                vertex_example = {
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}]
                    },
                    "contents": []
                }
                
                for msg in example.get("messages", []):
                    role = msg["role"]
                    if role == "system":
                        continue  # Already handled above
                    
                    vertex_role = "user" if role == "user" else "model"
                    vertex_example["contents"].append({
                        "role": vertex_role,
                        "parts": [{"text": msg["content"]}]
                    })
                
                if vertex_example["contents"]:
                    processed_examples.append(vertex_example)
        
        # Write processed data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for example in processed_examples:
                f.write(json.dumps(example) + "\n")
        
        logger.info("Prepared training data", num_examples=len(processed_examples))
        return output_file
    
    def upload_to_gcs(self, local_file: str, gcs_path: str) -> str:
        """Upload file to Google Cloud Storage."""
        # Parse bucket and blob path
        bucket_name = self.staging_bucket.replace("gs://", "").split("/")[0]
        blob_path = gcs_path.lstrip("/")
        
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        blob.upload_from_filename(local_file)
        gcs_uri = f"gs://{bucket_name}/{blob_path}"
        
        logger.info("Uploaded to GCS", local_file=local_file, gcs_uri=gcs_uri)
        return gcs_uri
    
    def start_tuning_job(
        self,
        training_data_uri: str,
        validation_data_uri: Optional[str] = None
    ) -> sft.SupervisedTuningJob:
        """
        Start a supervised fine-tuning job on Vertex AI.
        
        Args:
            training_data_uri: GCS URI of training data
            validation_data_uri: Optional GCS URI of validation data
            
        Returns:
            SupervisedTuningJob object
        """
        model_config = self.config["model"]
        training_config = self.config["training"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuned_model_name = f"{model_config['tuned_model_display_name']}_{timestamp}"
        
        logger.info(
            "Starting fine-tuning job",
            base_model=model_config["base_model"],
            tuned_model_name=tuned_model_name,
            epochs=training_config["epochs"]
        )
        
        # Create tuning job
        tuning_job = sft.train(
            source_model=model_config["base_model"],
            train_dataset=training_data_uri,
            validation_dataset=validation_data_uri,
            epochs=training_config["epochs"],
            adapter_size=training_config.get("adapter_size", "ADAPTER_SIZE_FOUR"),
            learning_rate_multiplier=training_config.get("learning_rate_multiplier", 1.0),
            tuned_model_display_name=tuned_model_name,
        )
        
        logger.info("Tuning job created", job_name=tuning_job.name)
        return tuning_job
    
    def monitor_job(self, job: sft.SupervisedTuningJob) -> None:
        """Monitor tuning job until completion."""
        logger.info("Monitoring tuning job...", job_name=job.name)
        
        # This blocks until the job completes
        job.wait()
        
        if job.has_ended:
            if job.has_succeeded:
                logger.info(
                    "Tuning job completed successfully!",
                    tuned_model=job.tuned_model_endpoint_name
                )
            else:
                logger.error("Tuning job failed", error=job.error)
    
    def test_tuned_model(self, model_endpoint: str, test_prompts: list[str]) -> None:
        """Test the fine-tuned model with sample prompts."""
        logger.info("Testing tuned model", endpoint=model_endpoint)
        
        model = GenerativeModel(model_endpoint)
        
        for prompt in test_prompts:
            logger.info("Testing prompt", prompt=prompt[:100])
            response = model.generate_content(prompt)
            logger.info("Response", response=response.text[:500])
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"Response: {response.text}")
            print(f"{'='*60}\n")


@click.command()
@click.option(
    "--config",
    default="configs/finetune_config.yaml",
    help="Path to configuration file"
)
@click.option(
    "--training-data",
    default="data/processed/training_data.jsonl",
    help="Path to training data JSONL"
)
@click.option(
    "--validation-data",
    default=None,
    help="Path to validation data JSONL (optional)"
)
@click.option(
    "--skip-upload",
    is_flag=True,
    help="Skip data upload (use existing GCS data)"
)
@click.option(
    "--monitor/--no-monitor",
    default=True,
    help="Monitor job until completion"
)
def main(
    config: str,
    training_data: str,
    validation_data: Optional[str],
    skip_upload: bool,
    monitor: bool
):
    """Run the AfriMed fine-tuning pipeline."""
    
    tuner = AfriMedFineTuner(config)
    
    if not skip_upload:
        # Prepare and upload training data
        processed_train = tuner.prepare_training_data(
            training_data,
            "data/processed/vertex_training.jsonl"
        )
        train_uri = tuner.upload_to_gcs(
            processed_train,
            f"training_data/train_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        
        # Handle validation data if provided
        val_uri = None
        if validation_data:
            processed_val = tuner.prepare_training_data(
                validation_data,
                "data/processed/vertex_validation.jsonl"
            )
            val_uri = tuner.upload_to_gcs(
                processed_val,
                f"training_data/val_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
    else:
        # Use existing GCS paths (you'd need to specify these)
        train_uri = f"{tuner.staging_bucket}/training_data/latest_train.jsonl"
        val_uri = None
    
    # Start tuning job
    job = tuner.start_tuning_job(train_uri, val_uri)
    
    if monitor:
        tuner.monitor_job(job)
        
        # Test the model if successful
        if job.has_succeeded:
            test_prompts = [
                "A pregnant woman in her third trimester has severe headache and blurred vision. What should I do?",
                "Mama mjamzito ana wiki 32 na ana maumivu ya tumbo. Je, ni kawaida?",
                "A newborn is 2 days old and has yellow skin. Is this normal?",
            ]
            tuner.test_tuned_model(job.tuned_model_endpoint_name, test_prompts)


if __name__ == "__main__":
    main()
