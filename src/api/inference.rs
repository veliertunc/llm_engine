// inference_api.rs
use crate::model::tokenizer::Tokenizer;
use crate::model::GLOBAL_MODEL;
use actix_web::{post, web, App, HttpResponse, HttpServer, Responder};

#[post("/infer")]
async fn infer_api(req_body: String) -> impl Responder {
    let tokens = simple_tokenizer(&req_body);
    let prediction = GLOBAL_MODEL.lock().unwrap().forward(&tokens);
    HttpResponse::Ok().json(prediction)
}

pub fn run_inference_server() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(infer_api))
        .bind("127.0.0.1:8080")?
        .run()
        .map_err(|e| e.into())
}
