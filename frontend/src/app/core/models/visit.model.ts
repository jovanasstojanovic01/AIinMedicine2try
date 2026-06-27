// visit.model.ts
export interface Visit {
  id?: number;
  patient_id: number;
  exam_date?: string;
  image_od_path?: string;
  mask_path?: string;
  // ostali parametri specifični za pregled
}
